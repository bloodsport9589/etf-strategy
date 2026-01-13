import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动量轮动策略 Pro+", page_icon="📈", layout="wide")

st.title("📈 全球资产动能轮动策略 Pro+")
st.markdown("### 趋势跟随 | 动能轮动 | 均线风控 | 深度回测")

# ================= 策略配置 =================
# 策略参数
HOLD_COUNT = 2          # 持仓数量
MOMENTUM_FAST = 20      # 20日涨幅
MOMENTUM_SLOW = 60      # 60日涨幅
MA_FILTER = 60          # 均线防守
BACKTEST_START = "20200101" 

# 交易标的池
ASSETS = {
    "513100": "纳指ETF",       # 美股
    "513520": "日经ETF",       # 日本
    "513180": "恒生科技",      # 港股
    "510180": "上证180",       # A股价值
    "159915": "创业板指",      # A股成长
    "518880": "黄金ETF",       # 商品避险
    "512400": "有色ETF",       # 周期
    "159981": "能源ETF",       # 能源
    "588050": "科创50",        # 硬科技
    "501018": "南方原油",      # 原油
}

# 额外的基准 (用于画图对比，不参与交易)
BENCHMARKS = {
    "510300": "沪深300"
}

# ================= 核心计算函数 =================

def calculate_max_drawdown(series):
    """计算最大回撤"""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def calculate_cagr(series):
    """计算年化收益率"""
    if len(series) < 1: return 0
    days = (series.index[-1] - series.index[0]).days
    if days == 0: return 0
    total_ret = series.iloc[-1] / series.iloc[0]
    return (total_ret) ** (365 / days) - 1

@st.cache_data(ttl=43200) 
def get_historical_data():
    """拉取所有数据 (交易标的 + 基准)"""
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 合并两个字典去拉取
    all_targets = {**ASSETS, **BENCHMARKS}
    
    progress_text = "正在拉取历史数据..."
    my_bar = st.progress(0, text=progress_text)
    total = len(all_targets)
    
    for i, (code, name) in enumerate(all_targets.items()):
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=BACKTEST_START, end_date=end_date, adjust="qfq")
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df[['close']]
            df.columns = [name]
            
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
        except Exception:
            pass
        my_bar.progress((i + 1) / total)
        
    my_bar.empty()
    return combined_df.sort_index().fillna(method='ffill')

def run_backtest(df_close):
    """回测引擎"""
    # 仅使用交易标的进行计算
    trade_assets = list(ASSETS.values())
    # 确保列存在
    available_assets = [c for c in trade_assets if c in df_close.columns]
    df_trade = df_close[available_assets]
    
    ret_daily = df_trade.pct_change()
    mom_20 = df_trade.pct_change(MOMENTUM_FAST)
    mom_60 = df_trade.pct_change(MOMENTUM_SLOW)
    score_df = mom_20 * 0.6 + mom_60 * 0.4
    ma_60 = df_trade.rolling(window=MA_FILTER).mean()
    
    strategy_curve = [1.0]
    dates = [df_trade.index[MA_FILTER]]
    start_idx = MA_FILTER
    pos_history = [] 

    for i in range(start_idx, len(df_trade) - 1):
        current_scores = score_df.iloc[i]
        current_prices = df_trade.iloc[i]
        current_ma = ma_60.iloc[i]
        
        trend_ok = current_prices > current_ma
        mom_ok = current_scores > 0
        valid_assets = current_scores[trend_ok & mom_ok]
        
        if not valid_assets.empty:
            targets = valid_assets.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
        else:
            targets = [] 
        
        next_day_ret_series = ret_daily.iloc[i+1]
        
        if len(targets) > 0:
            daily_pnl = next_day_ret_series[targets].mean()
            pos_history.append(",".join(targets))
        else:
            daily_pnl = 0.0 
            pos_history.append("现金")
            
        new_nav = strategy_curve[-1] * (1 + daily_pnl)
        strategy_curve.append(new_nav)
        dates.append(df_trade.index[i+1])
        
    return pd.Series(strategy_curve, index=dates), pos_history

# ================= 主程序 =================

df_all = get_historical_data()

if not df_all.empty:
    # 1. 运行策略回测
    strategy_nav, pos_history = run_backtest(df_all)
    
    # 2. 提取基准数据并归一化 (让起点都为1)
    common_start_date = strategy_nav.index[0]
    
    # 辅助函数：截取同时间段并归一化
    def get_normalized_benchmark(name):
        if name in df_all.columns:
            s = df_all[name].loc[common_start_date:]
            return s / s.iloc[0]
        return None

    bench_nasdaq = get_normalized_benchmark("纳指ETF")
    bench_nikkei = get_normalized_benchmark("日经ETF")
    bench_hs300 = get_normalized_benchmark("沪深300")

    # ================= 顶部 KPI 栏 =================
    st.subheader("📊 历史回测表现 (自 2020 年起)")
    
    # 计算指标
    strat_cagr = calculate_cagr(strategy_nav)
    strat_dd = calculate_max_drawdown(strategy_nav)
    strat_total = (strategy_nav.iloc[-1] - 1)
    
    # 沪深300指标对比
    hs300_cagr = calculate_cagr(bench_hs300) if bench_hs300 is not None else 0
    hs300_dd = calculate_max_drawdown(bench_hs300) if bench_hs300 is not None else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("策略总回报", f"{strat_total*100:.1f}%", help="策略至今的累计涨幅")
    kpi2.metric("年化回报 (CAGR)", f"{strat_cagr*100:.1f}%", delta=f"{(strat_cagr-hs300_cagr)*100:.1f}% vs 沪深300")
    kpi3.metric("最大回撤", f"{strat_dd*100:.1f}%", help="历史上最惨的一次跌幅")
    kpi4.metric("当前净值", f"{strategy_nav.iloc[-1]:.3f}")

    # ================= 中间 图表区 =================
    
    fig = go.Figure()

    # 画策略线 (最粗，高亮)
    fig.add_trace(go.Scatter(
        x=strategy_nav.index, y=strategy_nav,
        mode='lines', name='我的策略',
        line=dict(color='#00ff88', width=3) # 亮绿色
    ))

    # 画基准线 (细线，颜色区分)
    if bench_nasdaq is not None:
        fig.add_trace(go.Scatter(x=bench_nasdaq.index, y=bench_nasdaq, mode='lines', name='纳指ETF', line=dict(color='#3366ff', width=1.5, dash='dot')))
    
    if bench_nikkei is not None:
        fig.add_trace(go.Scatter(x=bench_nikkei.index, y=bench_nikkei, mode='lines', name='日经ETF', line=dict(color='#ff9900', width=1.5, dash='dot')))
        
    if bench_hs300 is not None:
        fig.add_trace(go.Scatter(x=bench_hs300.index, y=bench_hs300, mode='lines', name='沪深300', line=dict(color='#ff3333', width=1.5)))

    fig.update_layout(
        title="策略净值 vs 核心指数",
        xaxis_title="",
        yaxis_title="累计净值 (起点=1.0)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================= 底部 信号区 =================
    st.divider()
    
    # 准备今日数据
    latest_close = df_all.iloc[-1]
    trade_assets = list(ASSETS.values())
    valid_cols = [c for c in trade_assets if c in df_all.columns]
    
    # 重算今日动能
    df_trade = df_all[valid_cols]
    roc_20 = df_trade.pct_change(MOMENTUM_FAST).iloc[-1]
    roc_60 = df_trade.pct_change(MOMENTUM_SLOW).iloc[-1]
    ma_60 = df_trade.rolling(window=MA_FILTER).mean().iloc[-1]
    latest_score = roc_20 * 0.6 + roc_60 * 0.4
    
    rank_data = []
    for name in valid_cols:
        p = latest_close[name]
        m = ma_60[name]
        rank_data.append({
            "名称": name,
            "综合动能": latest_score[name],
            "现价": p,
            "60日趋势": "✅ 上涨" if p > m else "❌ 下跌",
            "20日涨幅": roc_20[name]
        })
    
    rank_df = pd.DataFrame(rank_data).sort_values(by="综合动能", ascending=False).reset_index(drop=True)
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("💡 明日操作建议")
        candidates = rank_df.head(HOLD_COUNT)
        buy_list = []
        for _, row in candidates.iterrows():
            if row['综合动能'] > 0 and row['60日趋势'] == "✅ 上涨":
                buy_list.append(row['名称'])
        
        if not buy_list:
            st.warning("🛑 **空仓/防守模式**：所有标的均走弱，建议持有现金/货基。")
        else:
            st.success(f"✅ **建议持仓**：\n\n **{' + '.join(buy_list)}**")
            if len(buy_list) < HOLD_COUNT:
                st.info("注：其余仓位建议现金。")

    with c2:
        st.subheader("📋 实时动能榜单")
        display_df = rank_df.head(5).copy()
        display_df['综合动能'] = display_df['综合动能'].apply(lambda x: f"{x*100:.2f}%")
        display_df['20日涨幅'] = display_df['20日涨幅'].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(display_df[['名称', '综合动能', '60日趋势', '20日涨幅']], use_container_width=True)

    with st.expander("查看最近调仓记录"):
        history_df = pd.DataFrame({
            "日期": strategy_nav.index[-10:], 
            "持仓": pos_history[-10:]
        }).sort_values("日期", ascending=False)
        st.table(history_df)

else:
    st.error("数据拉取失败，请刷新页面重试。")
