import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动量王者 (Top 1)", page_icon="👑", layout="wide")

st.title("👑 全球动量王者策略 (Winner Takes All)")
st.markdown("### 赢家通吃 | 单一持仓 | 挑战纳指 | 全面对比")

# ================= 策略配置 =================
# 核心参数：只持有一只！
HOLD_COUNT = 1          
MOMENTUM_WINDOW = 20    # 20日动能
MA_EXIT = 20            # 20日均线 (生命线，跌破空仓)
BACKTEST_START = "20200101" 

# 交易标的池 (你的弹药库)
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

# 基准池 (用于画图对比，不参与交易)
# 注意：纳指和日经已经在ASSETS里了，这里不需要重复拉取，在画图时直接用即可
# 这里只放不在交易池里的额外基准
BENCHMARKS_EXTRA = {
    "510300": "沪深300"
}

# ================= 核心逻辑 =================

def calculate_max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def calculate_cagr(series):
    if len(series) < 1: return 0
    days = (series.index[-1] - series.index[0]).days
    if days == 0: return 0
    return (series.iloc[-1] / series.iloc[0]) ** (365 / days) - 1

@st.cache_data(ttl=43200) 
def get_historical_data():
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 1. 拉取交易资产
    progress_bar = st.progress(0)
    total = len(ASSETS) + len(BENCHMARKS_EXTRA)
    current = 0
    
    # 拉取 ASSETS
    for code, name in ASSETS.items():
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=BACKTEST_START, end_date=end_date, adjust="qfq")
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')[['close']]
            df.columns = [name]
            
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
        except: pass
        current += 1
        progress_bar.progress(current / total)

    # 拉取额外基准 (沪深300)
    for code, name in BENCHMARKS_EXTRA.items():
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=BACKTEST_START, end_date=end_date, adjust="qfq")
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')[['close']]
            df.columns = [name]
            combined_df = combined_df.join(df, how='outer')
        except: pass
        current += 1
        progress_bar.progress(current / total)
    
    progress_bar.empty()
    return combined_df.sort_index().fillna(method='ffill')

def run_backtest(df_close):
    # 只选交易资产进行回测
    trade_assets = list(ASSETS.values())
    valid_cols = [c for c in trade_assets if c in df_close.columns]
    df_trade = df_close[valid_cols]
    
    ret_daily = df_trade.pct_change()
    score_df = df_trade.pct_change(MOMENTUM_WINDOW) # 只看20日爆发力
    ma_exit = df_trade.rolling(window=MA_EXIT).mean()   # MA20
    
    strategy_curve = [1.0]
    dates = [df_trade.index[MA_EXIT]]
    start_idx = MA_EXIT
    pos_history = [] 

    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        ma_short = ma_exit.iloc[i]
        
        # 1. 动能 > 0
        valid_assets = scores[scores > 0]
        
        # 2. 排序取 Top 1
        targets = []
        if not valid_assets.empty:
            targets = valid_assets.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
        
        # 3. 风控：必须在 MA20 之上
        final_holdings = []
        for asset in targets:
            if prices[asset] > ma_short[asset]:
                final_holdings.append(asset)
            # else: 即使你是第一名，如果跌破均线，也不买，直接空仓
        
        # 4. 计算收益
        daily_pnl = 0.0
        if len(final_holdings) > 0:
            # 全仓一只
            next_ret = ret_daily.iloc[i+1][final_holdings[0]]
            daily_pnl = next_ret
            pos_history.append(final_holdings[0])
        else:
            pos_history.append("现金")
            
        new_nav = strategy_curve[-1] * (1 + daily_pnl)
        strategy_curve.append(new_nav)
        dates.append(df_trade.index[i+1])

    return pd.Series(strategy_curve, index=dates), pos_history

# ================= 主程序 =================

df_all = get_historical_data()

if not df_all.empty:
    strategy_nav, pos_history = run_backtest(df_all)
    
    # 提取三大指数基准
    bench_nasdaq = df_all.get("纳指ETF")
    bench_nikkei = df_all.get("日经ETF")
    bench_hs300 = df_all.get("沪深300")
    
    start_date = strategy_nav.index[0]
    
    # 归一化函数
    def normalize(series):
        if series is not None:
            s = series.loc[start_date:]
            return s / s.iloc[0]
        return None

    bench_nasdaq_norm = normalize(bench_nasdaq)
    bench_nikkei_norm = normalize(bench_nikkei)
    bench_hs300_norm = normalize(bench_hs300)

    # --- KPI 区域 ---
    strat_cagr = calculate_cagr(strategy_nav)
    strat_dd = calculate_max_drawdown(strategy_nav)
    nasdaq_cagr = calculate_cagr(bench_nasdaq_norm) if bench_nasdaq_norm is not None else 0
    nasdaq_dd = calculate_max_drawdown(bench_nasdaq_norm) if bench_nasdaq_norm is not None else 0
    
    st.subheader("📊 巅峰对决 (策略 vs 纳指)")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👑 策略年化回报", f"{strat_cagr*100:.1f}%", delta=f"{(strat_cagr-nasdaq_cagr)*100:.1f}% vs 纳指")
    k2.metric("🛡️ 策略最大回撤", f"{strat_dd*100:.1f}%", help="该策略历史最大跌幅")
    k3.metric("📉 纳指最大回撤", f"{nasdaq_dd*100:.1f}%", delta_color="off")
    k4.metric("💰 当前净值", f"{strategy_nav.iloc[-1]:.3f}")

    # --- 核心图表 (4条线) ---
    fig = go.Figure()
    
    # 1. 策略线 (亮绿，最粗)
    fig.add_trace(go.Scatter(x=strategy_nav.index, y=strategy_nav, mode='lines', name='👑 Winner策略', line=dict(color='#00ff88', width=3)))
    
    # 2. 纳指 (蓝色，粗实线，作为主要对手)
    if bench_nasdaq_norm is not None:
        fig.add_trace(go.Scatter(x=bench_nasdaq_norm.index, y=bench_nasdaq_norm, mode='lines', name='纳指100', line=dict(color='#3366ff', width=2)))

    # 3. 日经 (橙色，虚线)
    if bench_nikkei_norm is not None:
        fig.add_trace(go.Scatter(x=bench_nikkei_norm.index, y=bench_nikkei_norm, mode='lines', name='日经225', line=dict(color='#ff9900', width=1.5, dash='dot')))

    # 4. 沪深300 (红色，虚线)
    if bench_hs300_norm is not None:
        fig.add_trace(go.Scatter(x=bench_hs300_norm.index, y=bench_hs300_norm, mode='lines', name='沪深300', line=dict(color='#ff3333', width=1.5, dash='dot')))

    fig.update_layout(
        template="plotly_dark", 
        hovermode="x unified", 
        title="全市场净值竞赛 (2020至今)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 信号区 ---
    st.divider()
    
    # 计算今日因子
    trade_df = df_all[list(ASSETS.values())]
    scores = trade_df.pct_change(MOMENTUM_WINDOW).iloc[-1]
    prices = trade_df.iloc[-1]
    ma_20 = trade_df.rolling(MA_EXIT).mean().iloc[-1]
