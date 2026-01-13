import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动能工厂 (全能版)", page_icon="🏭", layout="wide")

# ================= 侧边栏：参数控制区 =================
st.sidebar.header("🎛️ 策略控制台")

# 1. 动能因子构造
st.sidebar.subheader("1. 动能因子 (ROC)")
ROC_SHORT = st.sidebar.slider("短期 ROC 周期 (天)", 5, 60, 20)
ROC_LONG = st.sidebar.slider("长期 ROC 周期 (天)", 30, 250, 60)
_weight_raw = st.sidebar.slider("短期权重 (%)", 0, 100, 100)
ROC_WEIGHT = _weight_raw / 100.0

# 2. 交易参数
st.sidebar.subheader("2. 交易执行")
HOLD_COUNT = st.sidebar.number_input("持仓数量 (Top N)", min_value=1, max_value=5, value=1)
MA_EXIT = st.sidebar.number_input("止损均线 (MA)", min_value=5, max_value=120, value=20, help="生命线，跌破即空仓")

# 3. 回测设置
BACKTEST_START = st.sidebar.date_input("回测开始日期", datetime.date(2020, 1, 1))

# 标的池
ASSETS = {
    "513100": "纳指ETF",       
    "513520": "日经ETF",       
    "513180": "恒生科技",      
    "510180": "上证180",       
    "159915": "创业板指",      
    "518880": "黄金ETF",       
    "512400": "有色ETF",       
    "159981": "能源ETF",       
    "588050": "科创50",        
    "501018": "南方原油",      
}
# 额外基准 (沪深300)
BENCHMARKS = {"510300": "沪深300"}

# ================= 核心计算逻辑 =================

@st.cache_data(ttl=43200) 
def get_historical_data(start_date):
    """获取数据"""
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")
    
    # 合并拉取 ASSETS 和 BENCHMARKS
    targets = {**ASSETS, **BENCHMARKS}
    progress = st.empty()
    
    for i, (code, name) in enumerate(targets.items()):
        progress.text(f"正在加载: {name}...")
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date, adjust="qfq")
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')[['close']]
            df.columns = [name]
            
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
        except: pass
    
    progress.empty()
    return combined_df.sort_index().fillna(method='ffill')

def calculate_factors(df, roc_s, roc_l, w_s):
    """计算复合因子"""
    # 只针对交易标的计算因子
    trade_cols = list(ASSETS.values())
    # 过滤掉不在df里的列(防报错)
    valid_cols = [c for c in trade_cols if c in df.columns]
    df_trade = df[valid_cols]
    
    roc_short = df_trade.pct_change(roc_s)
    roc_long = df_trade.pct_change(roc_l)
    
    score = roc_short * w_s + roc_long * (1 - w_s)
    ma_exit = df_trade.rolling(MA_EXIT).mean()
    
    return score, ma_exit, df_trade

def run_backtest(df_trade, score_df, ma_df):
    """回测引擎"""
    start_idx = max(ROC_LONG, ROC_SHORT, MA_EXIT) + 1
    if start_idx >= len(df_trade): return None, None, None
    
    curve = [1.0]
    dates = [df_trade.index[start_idx]]
    pos_history = []
    
    ret_daily = df_trade.pct_change()
    factor_analysis_data = [] 

    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        mas = ma_df.iloc[i]
        
        # --- 1. 交易逻辑 ---
        valid = scores[(scores > 0) & (prices > mas)]
        
        targets = []
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
            
        day_pnl = 0.0
        if targets:
            w = 1.0 / HOLD_COUNT 
            rets = ret_daily.iloc[i+1][targets]
            day_pnl = rets.sum() * w
            pos_history.append(",".join(targets))
        else:
            pos_history.append("现金")
            
        curve.append(curve[-1] * (1 + day_pnl))
        dates.append(df_trade.index[i+1])
        
        # --- 2. 因子有效性数据收集 ---
        # 每天都记录：[排名, 次日收益]
        daily_rank = scores.rank(ascending=False, method='first') 
        next_day_ret = ret_daily.iloc[i+1]
        
        for asset in scores.index:
            r = daily_rank.get(asset)
            ret = next_day_ret.get(asset)
            # 确保数据不是NaN
            if pd.notnull(r) and pd.notnull(ret):
                factor_analysis_data.append({
                    "Rank": int(r),
                    "Return": ret
                })

    return pd.Series(curve, index=dates), pos_history, pd.DataFrame(factor_analysis_data)

# ================= 主界面 =================

st.title("🏭 动能策略工厂 (全能版)")
st.markdown("集成 **实时信号**、**多指数对比回测** 与 **因子有效性体检**。")

df_all = get_historical_data(BACKTEST_START)

if not df_all.empty:
    # 计算因子
    score_df, ma_df, df_trade = calculate_factors(df_all, ROC_SHORT, ROC_LONG, ROC_WEIGHT)
    nav, history, factor_data = run_backtest(df_trade, score_df, ma_df)
    
    if nav is not None:
        
        # ==========================================
        # 💡 Part 1: 今日实盘信号
        # ==========================================
        st.divider()
        st.header("💡 今日实盘信号")
        
        latest_scores = score_df.iloc[-1]
        latest_prices = df_trade.iloc[-1]
        latest_mas = ma_df.iloc[-1]
        data_date = score_df.index[-1].strftime('%Y-%m-%d')
        
        st.caption(f"数据日期: {data_date}")
        
        rank_data = []
        for name in latest_scores.index:
            s = latest_scores.get(name, -99)
            p = latest_prices.get(name, 0)
            m = latest_mas.get(name, 0)
            is_buy = (s > 0) and (p > m)
            rank_data.append({
                "名称": name,
                "综合动能": s,
                "现价": p,
                "止损线": m,
                "状态": "✅ 持有" if is_buy else "❌ 空仓"
            })
            
        df_rank = pd.DataFrame(rank_data).sort_values("综合动能", ascending=False).reset_index(drop=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("📢 操作建议")
            buys = df_rank[df_rank['状态'] == "✅ 持有"].head(HOLD_COUNT)
            if buys.empty:
                st.error("🛑 **空仓信号** (持有现金)")
            else:
                st.success("✅ **买入/持有**")
                for _, row in buys.iterrows():
                    st.write(f"**{row['名称']}**")
        
        with c2:
            st.subheader("📊 实时排行榜")
            d_show = df_rank.copy()
            d_show['综合动能'] = d_show['综合动能'].apply(lambda x: f"{x*100:.2f}%")
            d_show['止损线'] = d_show['止损线'].apply(lambda x: f"{x:.3f}")
            def color_status(v):
                return f'color: {"#00ff88" if "✅" in v else "#ff4444"}; font-weight: bold'
            st.dataframe(d_show.style.applymap(color_status, subset=['状态']), use_container_width=True)

        st.divider()

        # ==========================================
        # 💡 Part 2: 分析 Tabs
        # ==========================================
        tab1, tab2 = st.tabs(["📈 策略回测 (多指数对比)", "🔬 因子有效性体检"])
        
        # ----- Tab 1: 回测图表 -----
        with tab1:
            # 准备基准数据 (归一化)
            start_dt = nav.index[0]
            
            def get_norm_bench(name):
                if name in df_all.columns:
                    s = df_all[name].loc[start_dt:]
                    if not s.empty:
                        return s / s.iloc[0]
                return None

            b_nasdaq = get_norm_bench("纳指ETF")
            b_nikkei = get_norm_bench("日经ETF")
            b_hs300 = get_norm_bench("沪深300")
            
            # 业绩指标
            total_ret = (nav.iloc[-1] - 1) * 100
            days = (nav.index[-1] - nav.index[0]).days
            cagr = (nav.iloc[-1] ** (365 / days) - 1) * 100 if days > 0 else 0
            drawdown = ((nav - nav.cummax()) / nav.cummax()).min() * 100
            
            n_ret = (b_nasdaq.iloc[-1]-1)*100 if b_nasdaq is not None else 0
            
            st.write("### 核心业绩")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益率", f"{total_ret:.1f}%", delta=f"{total_ret - n_ret:.1f}% vs 纳指")
            k2.metric("年化收益", f"{cagr:.1f}%")
            k3.metric("最大回撤", f"{drawdown:.1f}%")
            k4.metric("策略参数", f"ROC: {int(ROC_SHORT)}/{int(ROC_LONG)} | MA: {MA_EXIT}")

            # 绘图
            fig = go.Figure()
            # 1. 策略 (绿, 粗)
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name='策略净值', line=dict(color='#00ff88', width=3)))
            # 2. 纳指 (蓝)
            if b_nasdaq is not None:
                fig.add_trace(go.Scatter(x=b_nasdaq.index, y=b_nasdaq, name='纳指100', line=dict(color='#3366ff', width=1.5)))
            # 3. 日经 (橙)
            if b_nikkei is not None:
                fig.add_trace(go.Scatter(x=b_nikkei.index, y=b_nikkei, name='日经225', line=dict(color='#ff9900', width=1.5, dash='dot')))
            # 4. 沪深300 (红)
            if b_hs300 is not None:
                fig.add_trace(go.Scatter(x=b_hs300.index, y=b_hs300, name='沪深300', line=dict(color='#ff3333', width=1.5, dash='dot')))
                
            fig.update_layout(template="plotly_dark", title="全球核心资产净值竞赛", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # ----- Tab 2: 因子体检 -----
        with tab2:
            st.write("### 🔬 动能因子有效性检验")
            st.info("原理：统计过去每天按动能排名（Rank 1为最强）的品种，在次日的平均涨跌幅。如果柱状图呈阶梯下降，说明因子有效。")
            
            if not factor_data.empty:
                # 分组计算平均收益
                rank_perf = factor_data.groupby("Rank")["Return"].mean() * 100 
                
                # 画图
                fig_bar = px.bar(
                    x=rank_perf.index, 
                    y=rank_perf.values,
                    labels={'x': '动能排名 (1=最强)', 'y': '次日平均涨幅 (%)'},
                    title=f"分层回测 (样本: {len(factor_data)} 个交易日)",
                    color=rank_perf.values,
                    color_continuous_scale="RdYlGn",
                    text_auto='.3f'
                )
                fig_bar.update_layout(template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 结论分析
                top1_ret = rank_perf.get(1, 0)
                bottom_ret = rank_perf.iloc[-1]
                diff = top1_ret - bottom_ret
                
                c1, c2 = st.columns(2)
                c1.metric("Top 1 平均日收益", f"{top1_ret:.3f}%")
                c2.metric("多空收益差 (Top - Bottom)", f"{diff:.3f}%")
                
                if diff > 0.05:
                    st.success("✅ **强有效**：排名第一的显著跑赢倒数第一。")
                elif diff > 0:
                    st.warning("⚠️ **弱有效**：区分度一般，可能需要调整 ROC 参数。")
                else:
                    st.error("🛑 **失效/反转**：排名靠前的反而亏损，动能因子在当前参数下失效。")
            else:
                st.warning("数据不足，无法生成体检报告。请检查回测时间范围。")

    else:
        st.error("无法计算策略净值，请检查数据源或参数设置。")
else:
    st.error("数据加载失败。")
