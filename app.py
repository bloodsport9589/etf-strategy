import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动量策略 (进攻版)", page_icon="🚀", layout="wide")

st.title("🚀 全球动量策略 v3.0 (极速趋势版)")
st.markdown("### 逻辑升级：纯粹动能 | 双均线风控 | 跌破MA20极速离场")

# ================= 策略配置 =================
# 核心参数
HOLD_COUNT = 2          # 持仓数量
MOMENTUM_WINDOW = 20    # 动能窗口 (只看20日爆发力)
MA_ENTRY = 60           # 进场趋势线 (牛熊分界)
MA_EXIT = 20            # 离场生命线 (跌破即跑)
BACKTEST_START = "20200101" 

# 资产池
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

# ================= 计算核心 =================

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
    all_targets = {**ASSETS, **BENCHMARKS}
    
    progress_bar = st.progress(0)
    total = len(all_targets)
    
    for i, (code, name) in enumerate(all_targets.items()):
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
        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()
    return combined_df.sort_index().fillna(method='ffill')

def run_backtest(df_close):
    trade_assets = list(ASSETS.values())
    valid_cols = [c for c in trade_assets if c in df_close.columns]
    df_trade = df_close[valid_cols]
    
    # 1. 计算指标
    ret_daily = df_trade.pct_change()
    
    # 纯动能：只看 ROC 20
    score_df = df_trade.pct_change(MOMENTUM_WINDOW)
    
    # 双均线
    ma_entry = df_trade.rolling(window=MA_ENTRY).mean() # MA60
    ma_exit = df_trade.rolling(window=MA_EXIT).mean()   # MA20
    
    # 2. 回测循环
    strategy_curve = [1.0]
    dates = [df_trade.index[MA_ENTRY]]
    start_idx = MA_ENTRY
    pos_history = [] 

    for i in range(start_idx, len(df_trade) - 1):
        # 当日数据
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        ma_long = ma_entry.iloc[i]  # 60日线
        ma_short = ma_exit.iloc[i]  # 20日线
        
        # --- 选股逻辑 ---
        # 1. 动能必须 > 0
        valid_assets = scores[scores > 0]
        
        # 2. 排序取 Top N
        targets = []
        if not valid_assets.empty:
            targets = valid_assets.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
        
        # --- 风控逻辑 (Critical!) ---
        # 即使选进了 Top 2，如果当前价格跌破 MA20，强制把这部分仓位变成现金
        final_holdings = []
        
        for asset in targets:
            # 规则：
            # 如果是新开仓，必须站上 MA60 (牛市确认)
            # 如果是持仓中，只要站上 MA20 (趋势未坏) 即可持有
            # 这里简化为：只要在 Top 2 且 > MA20 就持有。
            # 为什么用 MA20？因为 MA60 反应太慢，MA20 能在暴跌初期止损。
            
            if prices[asset] > ma_short[asset]:
                final_holdings.append(asset)
            # else: 价格 < MA20，虽然动能强（可能是刚开始跌），但也强制空仓
            
        # 计算次日收益 (等权重)
        daily_pnl = 0.0
        
        # 假设总是把资金分成 HOLD_COUNT 份 (例如2份)
        # 如果 final_holdings 只有 1 个，那就是 50% 仓位，剩下 50% 现金
        if len(final_holdings) > 0:
            weight_per_asset = 1.0 / HOLD_COUNT 
            next_ret = ret_daily.iloc[i+1][final_holdings]
            daily_pnl = (next_ret * weight_per_asset).sum()
            
            pos_history.append(" + ".join(final_holdings))
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
    
    # 基准处理
    bench_nasdaq = df_all.get("纳指ETF")
    start_date = strategy_nav.index[0]
    
    if bench_nasdaq is not None:
        bench_nasdaq = bench_nasdaq.loc[start_date:]
        bench_nasdaq = bench_nasdaq / bench_nasdaq.iloc[0]

    # --- KPI ---
    strat_cagr = calculate_cagr(strategy_nav)
    strat_dd = calculate_max_drawdown(strategy_nav)
    nasdaq_cagr = calculate_cagr(bench_nasdaq) if bench_nasdaq is not None else 0
    nasdaq_dd = calculate_max_drawdown(bench_nasdaq) if bench_nasdaq is not None else 0
    
    st.subheader("📊 策略性能评估 (vs 纳指)")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("策略年化 (CAGR)", f"{strat_cagr*100:.1f}%", delta=f"{(strat_cagr-nasdaq_cagr)*100:.1f}% vs 纳指")
    k2.metric("最大回撤", f"{strat_dd*100:.1f}%", help="回撤越小越安全")
    k3.metric("纳指最大回撤", f"{nasdaq_dd*100:.1f}%", delta_color="off")
    k4.metric("当前净值", f"{strategy_nav.iloc[-1]:.3f}")

    # --- 图表 ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_nav.index, y=strategy_nav, mode='lines', name='进攻策略 v3', line=dict(color='#00ff88', width=2.5)))
    if bench_nasdaq is not None:
        fig.add_trace(go.Scatter(x=bench_nasdaq.index, y=bench_nasdaq, mode='lines', name='纳斯达克100', line=dict(color='#3366ff', width=1.5, dash='dot')))
    
    fig.update_layout(template="plotly_dark", hovermode="x unified", title="净值曲线对比")
    st.plotly_chart(fig, use_container_width=True)

    # --- 今日信号 ---
    st.divider()
    latest_date = df_all.index[-1]
    
    trade_df = df_all[list(ASSETS.values())]
    scores = trade_df.pct_change(MOMENTUM_WINDOW).iloc[-1]
    prices = trade_df.iloc[-1]
    ma_60 = trade_df.rolling(MA_ENTRY).mean().iloc[-1]
    ma_20 = trade_df.rolling(MA_EXIT).mean().iloc[-1]
    
    rank_data = []
    for name in ASSETS.values():
        if name in scores:
            rank_data.append({
                "名称": name,
                "20日涨幅": scores[name],
                "现价": prices[name],
                "MA20(止损线)": ma_20[name],
                "MA60(牛熊线)": ma_60[name],
                "状态": "✅" if (prices[name] > ma_20[name] and scores[name] > 0) else "❌"
            })
            
    rank_df = pd.DataFrame(rank_data).sort_values("20日涨幅", ascending=False).reset_index(drop=True)
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("💡 明日操作建议")
        # 选取 Top 2
        candidates = rank_df.head(HOLD_COUNT)
        
        buy_list = []
        for _, row in candidates.iterrows():
            if row['状态'] == '✅':
                buy_list.append(row['名称'])
        
        if not buy_list:
            st.warning("🛑 **全仓防守**：市场所有头部资产均跌破 MA20。")
        else:
            st.success("✅ **持有/买入**")
            for item in buy_list:
                st.write(f"**{item}** (仓位 50%)")
            
            if len(buy_list) < HOLD_COUNT:
                st.info(f"注：剩余 {50 * (HOLD_COUNT - len(buy_list))}% 仓位保持现金。")

    with c2:
        st.subheader("📋 实时排名 & 均线监控")
        # 格式化
        d_df = rank_df.copy()
        d_df['20日涨幅'] = d_df['20日涨幅'].apply(lambda x: f"{x*100:.2f}%")
        d_df['MA20(止损线)'] = d_df['MA20(止损线)'].apply(lambda x: f"{x:.3f}")
        
        def highlight_status(val):
            return 'color: #00ff88' if val == '✅' else 'color: #ff4444'
            
        st.dataframe(d_df.style.applymap(highlight_status, subset=['状态']), use_container_width=True)
