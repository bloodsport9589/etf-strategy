import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动能工厂 (实战版)", page_icon="🏭", layout="wide")

# ================= 侧边栏：参数控制区 =================
st.sidebar.header("🎛️ 策略控制台")

# 1. 动能因子构造 (ROC Parameter)
st.sidebar.subheader("1. 动能因子 (ROC)")
ROC_SHORT = st.sidebar.slider("短期 ROC 周期 (天)", 5, 60, 20)
ROC_LONG = st.sidebar.slider("长期 ROC 周期 (天)", 30, 250, 60)
# 权重计算
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
BENCHMARKS = {"510300": "沪深300"}

# ================= 核心计算逻辑 =================

@st.cache_data(ttl=43200) 
def get_historical_data(start_date):
    """获取数据"""
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")
    
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
    trade_cols = list(ASSETS.values())
    df_trade = df[trade_cols]
    
    # 计算双频 ROC
    roc_short = df_trade.pct_change(roc_s)
    roc_long = df_trade.pct_change(roc_l)
    
    # 合成得分
    score = roc_short * w_s + roc_long * (1 - w_s)
    
    # 均线
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
    
    # 用于有效性分析的数据
    factor_analysis_data = [] 

    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        mas = ma_df.iloc[i]
        
        # --- 1. 交易逻辑 ---
        # 选出动能 > 0 且 价格 > 均线 的
        valid = scores[(scores > 0) & (prices > mas)]
        
        targets = []
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
            
        # 计算次日收益
        day_pnl = 0.0
        if targets:
            w = 1.0 / HOLD_COUNT # 简单等权
            rets = ret_daily.iloc[i+1][targets]
            day_pnl = rets.sum() * w
            pos_history.append(",".join(targets))
        else:
            pos_history.append("现金")
            
        curve.append(curve[-1] * (1 + day_pnl))
        dates.append(df_trade.index[i+1])
        
        # --- 2. 收集数据用于因子有效性分析 ---
        daily_rank = scores.rank(ascending=False, method='first') 
        next_day_ret = ret_daily.iloc[i+1]
        
        for asset in scores.index:
            if not np.isnan(scores[asset]) and not np.isnan(next_day_ret[asset]):
                factor_analysis_data.append({
                    "Rank": int(daily_rank[asset]),
                    "Return": next_day_ret[asset]
                })

    return pd.Series(curve, index=dates), pos_history, pd.DataFrame(factor_analysis_data)

# ================= 主界面 =================

st.title("🏭 动能策略工厂 (实战版)")
st.markdown("通过调节参数优化策略，并提供**实时交易信号**。")

df_all = get_historical_data(BACKTEST_START)

if not df_all.empty:
    # 1. 计算
    score_df, ma_df, df_trade = calculate_factors(df_all, ROC_SHORT, ROC_LONG, ROC_WEIGHT)
    nav, history, factor_data = run_backtest(df_trade, score_df, ma_df)
    
    if nav is not None:
        
        # ==========================================
        # 💡 新增/恢复：今日实盘信号区 (放在最显眼的位置)
        # ==========================================
        st.divider()
        st.header("💡 今日实盘信号 (Real-time Signals)")
        
        # 获取最新一行数据
        latest_scores = score_df.iloc[-1]
        latest_prices = df_trade.iloc[-1]
        latest_mas = ma_df.iloc[-1]
        data_date = score_df.index[-1].strftime('%Y-%m-%d')
        
        st.caption(f"数据更新日期: {data_date} (请确保这是最新交易日)")
        
        # 构建当前状态表
        rank_data = []
        for name in latest_scores.index:
            s = latest_scores[name]
            p = latest_prices[name]
            m = latest_mas[name]
            
            # 状态判断: 动能>0 且 价格>均线
            is_buy = (s > 0) and (p > m)
            
            rank_data.append({
                "名称": name,
                "综合动能": s,
                "现价": p,
                "均线(止损)": m,
                "状态": "✅ 持有" if is_buy else "❌ 空仓"
            })
            
        df_rank = pd.DataFrame(rank_data)
        df_rank = df_rank.sort_values("综合动能", ascending=False).reset_index(drop=True)
        
        # 布局：左边是建议，右边是详细表格
        col_sig1, col_sig2 = st.columns([1, 2])
        
        with col_sig1:
            st.subheader("📢 操作建议")
            # 选出符合条件的 Top N
            valid_buys = df_rank[df_rank['状态'] == "✅ 持有"].head(HOLD_COUNT)
            
            if valid_buys.empty:
                st.error("🛑 **空仓信号**")
                st.write("所有资产均未触发买入条件（动能为负 或 跌破均线）。建议持有现金。")
            else:
                st.success("✅ **买入/持有列表**")
                for _, row in valid_buys.iterrows():
                    st.write(f"**{row['名称']}**")
                    st.caption(f"动能: {row['综合动能']*100:.2f}% | 离均线: {(row['现价']/row['均线(止损)']-1)*100:.1f}%")
                
                if len(valid_buys) < HOLD_COUNT:
                    st.info(f"注：仅 {len(valid_buys)} 只符合条件，其余仓位现金。")

        with col_sig2:
            st.subheader("📊 实时排行榜")
            # 格式化显示
            display_df = df_rank.copy()
            display_df['综合动能'] = display_df['综合动能'].apply(lambda x: f"{x*100:.2f}%")
            display_df['均线(止损)'] = display_df['均线(止损)'].apply(lambda x: f"{x:.3f}")
            
            # 高亮样式
            def highlight_signal(val):
                color = '#00ff88' if '✅' in val else '#ff4444'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(display_df.style.applymap(highlight_signal, subset=['状态']), use_container_width=True)

        st.divider()

        # ==========================================
        # 下面是之前的分析图表 (Tabs)
        # ==========================================
        
        tab1, tab2 = st.tabs(["📈 策略回测", "🔬 因子有效性体检"])
        
        with tab1:
            # 计算指标
            total_ret = (nav.iloc[-1] - 1) * 100
            days = (nav.index[-1] - nav.index[0]).days
            cagr = (nav.iloc[-1] ** (365 / days) - 1) * 100 if days > 0 else 0
            drawdown = ((nav - nav.cummax()) / nav.cummax()).min() * 100
            
            if '纳指ETF' in df_all.columns:
                nasdaq = df_all['纳指ETF'].loc[nav.index[0]:]
                nasdaq = nasdaq / nasdaq.iloc[0]
                nasdaq_ret = (nasdaq.iloc[-1] - 1) * 100
            else:
                nasdaq_ret = 0
                nasdaq = pd.Series()
            
            st.write("### 核心业绩")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益率", f"{total_ret:.1f}%", delta=f"{total_ret - nasdaq_ret:.1f}% vs 纳指")
            k2.metric("年化收益", f"{cagr:.1f}%")
            k3.metric("最大回撤", f"{drawdown:.1f}%")
            
            param_str = f"ROC: {int(ROC_SHORT)}日({int(ROC_WEIGHT*100)}%) + {int(ROC_LONG)}日"
            k4.metric("参数配置", param_str)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name='策略净值', line=dict(color='#00ff88', width=2)))
            if not nasdaq.empty:
                fig.add_trace(go.Scatter(x=nasdaq.index, y=nasdaq, name='纳指ETF', line=dict(color='#3366ff', width=1)))
            fig.update_layout(template="plotly_dark", title="净值曲线", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("### 🔬 动能因子有效性检验")
            
            if not factor_data.empty:
                rank_perf = factor_data.groupby("Rank")["Return"].mean() * 100 
                
                fig_bar = px.bar(
                    x=rank_perf.index, 
                    y=rank_perf.values,
                    labels={'x': '动能排名 (1=最强)', 'y': '次日平均涨幅 (%)'},
                    title=f"分层回测 (样本: {len(factor_data)}天)",
                    color=rank_perf.values,
                    color_continuous_scale="RdYlGn"
                )
                fig_bar.update_layout(template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
                
                top1_ret = rank_perf.get(1, 0)
                last_ret = rank_perf.iloc[-1]
                diff = top1_ret - last_ret
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Top 1 平均日收益", f"{top1_ret:.3f}%")
                with c2:
                    st.metric("多空收益差", f"{diff:.3f}%")
            else:
                st.write("数据不足以进行分析。")

    else:
        st.error("请调整回测时间，或检查数据源。")
