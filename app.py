import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动量轮动 Pro Max", page_icon="🛡️", layout="wide")

st.title("🛡️ 全球动量轮动 Pro Max (低波增强版)")
st.markdown("### 动能轮动 | 波动率加权 | RSI过热过滤 | 移动止损")

# ================= 策略配置 =================
# 核心参数
HOLD_COUNT = 2          # 持仓数量
MOMENTUM_FAST = 20      # 20日涨幅
MOMENTUM_SLOW = 60      # 60日涨幅
MA_FILTER = 60          # 趋势均线
RSI_WINDOW = 14         # RSI 周期
RSI_LIMIT = 82          # RSI 超买阈值 (超过这个不买/减仓)
VOL_WINDOW = 20         # 波动率计算周期
STOP_LOSS_PCT = 0.08    # 移动止损 (从最高点回撤 8% 离场)
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

# ================= 辅助计算函数 =================

def calculate_rsi(series, period=14):
    """计算 RSI 指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_volatility(df, window=20):
    """计算滚动波动率 (标准差)"""
    return df.pct_change().rolling(window=window).std()

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
    
    # 进度条
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

# ================= 核心回测引擎 (优化版) =================

def run_backtest(df_close):
    trade_assets = list(ASSETS.values())
    valid_cols = [c for c in trade_assets if c in df_close.columns]
    df_trade = df_close[valid_cols]
    
    # 1. 计算所有指标
    ret_daily = df_trade.pct_change()
    
    # 动能
    mom_20 = df_trade.pct_change(MOMENTUM_FAST)
    mom_60 = df_trade.pct_change(MOMENTUM_SLOW)
    score_df = mom_20 * 0.6 + mom_60 * 0.4
    
    # 均线
    ma_60 = df_trade.rolling(window=MA_FILTER).mean()
    
    # 波动率 (用于加权)
    vol_df = calculate_volatility(df_trade, VOL_WINDOW)
    
    # RSI (用于过滤)
    rsi_df = df_trade.apply(lambda x: calculate_rsi(x, RSI_WINDOW))
    
    # 2. 循环回测
    strategy_curve = [1.0]
    dates = [df_trade.index[MA_FILTER]]
    start_idx = MA_FILTER
    pos_history = [] 
    
    # 记录每个持有资产的最高价 (用于移动止损)
    high_water_mark = {asset: 0 for asset in valid_cols}
    current_holdings = []

    for i in range(start_idx, len(df_trade) - 1):
        today = df_trade.index[i]
        
        # 获取当日数据
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        mas = ma_60.iloc[i]
        rsis = rsi_df.iloc[i]
        vols = vol_df.iloc[i]
        
        # --- 筛选逻辑 ---
        # 1. 均线过滤
        cond_trend = prices > mas
        # 2. 动能过滤
        cond_mom = scores > 0
        # 3. RSI 过滤 (不能太热)
        cond_rsi = rsis < RSI_LIMIT
        
        valid_assets = scores[cond_trend & cond_mom & cond_rsi]
        
        # 排序选出 Top N
        targets = []
        if not valid_assets.empty:
            targets = valid_assets.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
        
        # --- 移动止损检查 ---
        # 如果某个资产本来在 targets 里，但触发了硬止损，把它剔除
        final_targets = []
        for asset in targets:
            # 更新最高水位线
            if asset not in current_holdings:
                high_water_mark[asset] = prices[asset] # 新买入，重置最高价
            else:
                high_water_mark[asset] = max(high_water_mark[asset], prices[asset])
            
            # 检查回撤
            drawdown = (prices[asset] - high_water_mark[asset]) / high_water_mark[asset]
            
            if drawdown > -STOP_LOSS_PCT: # 没有跌破 8%
                final_targets.append(asset)
            # else: 触发止损，不加入 final_targets (相当于卖出)

        current_holdings = final_targets
        
        # --- 波动率加权分配 ---
        # 如果选出2个，不是各50%，而是波动率越低给越多权重
        daily_pnl = 0.0
        
        if len(final_targets) > 0:
            target_vols = vols[final_targets]
            # 倒数加权: 1/vol
            inv_vols = 1 / (target_vols + 0.0001) # 防止除以0
            weights = inv_vols / inv_vols.sum()
            
            # 计算次日收益
            next_ret = ret_daily.iloc[i+1][final_targets]
            daily_pnl = (next_ret * weights).sum()
            
            # 记录历史 (带权重显示)
            pos_str = " | ".join([f"{t}({w:.0%})" for t, w in weights.items()])
            pos_history.append(pos_str)
        else:
            daily_pnl = 0.0
            pos_history.append("现金/避险")
            
        new_nav = strategy_curve[-1] * (1 + daily_pnl)
        strategy_curve.append(new_nav)
        dates.append(df_trade.index[i+1])

    return pd.Series(strategy_curve, index=dates), pos_history

# ================= 主程序逻辑 =================

df_all = get_historical_data()

if not df_all.empty:
    strategy_nav, pos_history = run_backtest(df_all)
    
    # 处理基准
    bench_nasdaq = df_all.get("纳指ETF")
    bench_hs300 = df_all.get("沪深300")
    
    # 归一化
    start_date = strategy_nav.index[0]
    if bench_nasdaq is not None: 
        bench_nasdaq = bench_nasdaq.loc[start_date:] 
        bench_nasdaq = bench_nasdaq / bench_nasdaq.iloc[0]
    if bench_hs300 is not None: 
        bench_hs300 = bench_hs300.loc[start_date:]
        bench_hs300 = bench_hs300 / bench_hs300.iloc[0]

    # --- KPI 显示 ---
    strat_cagr = calculate_cagr(strategy_nav)
    strat_dd = calculate_max_drawdown(strategy_nav)
    nasdaq_cagr = calculate_cagr(bench_nasdaq) if bench_nasdaq is not None else 0
    nasdaq_dd = calculate_max_drawdown(bench_nasdaq) if bench_nasdaq is not None else 0
    
    st.subheader("📊 策略性能评估")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("策略年化回报", f"{strat_cagr*100:.1f}%", help="Compound Annual Growth Rate")
    c2.metric("策略最大回撤", f"{strat_dd*100:.1f}%", delta=f"{-(nasdaq_dd - strat_dd)*100:.1f}% vs 纳指", delta_color="inverse", help="越小越好")
    c3.metric("收益回撤比 (Calmar)", f"{abs(strat_cagr/strat_dd):.2f}", help="衡量性价比，越高越好。通常 > 1.0 算优秀")
    c4.metric("当前净值", f"{strategy_nav.iloc[-1]:.3f}")

    # --- 绘图 ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_nav.index, y=strategy_nav, mode='lines', name='优化策略 (低波)', line=dict(color='#00ff88', width=3)))
    if bench_nasdaq is not None:
        fig.add_trace(go.Scatter(x=bench_nasdaq.index, y=bench_nasdaq, mode='lines', name='纳指ETF (基准)', line=dict(color='#3366ff', width=1, dash='dot')))
    if bench_hs300 is not None:
        fig.add_trace(go.Scatter(x=bench_hs300.index, y=bench_hs300, mode='lines', name='沪深300', line=dict(color='#ff3333', width=1)))

    fig.update_layout(title="策略 vs 基准 (引入波动率控制后)", template="plotly_dark", hovermode="x unified", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # --- 信号区 ---
    st.divider()
    latest_date = df_all.index[-1]
    
    # 重新计算最新一天的因子以展示
    trade_df = df_all[list(ASSETS.values())]
    
    mom_fast = trade_df.pct_change(MOMENTUM_FAST).iloc[-1]
    mom_slow = trade_df.pct_change(MOMENTUM_SLOW).iloc[-1]
    scores = mom_fast * 0.6 + mom_slow * 0.4
    
    mas = trade_df.rolling(MA_FILTER).mean().iloc[-1]
    rsis = trade_df.apply(lambda x: calculate_rsi(x, RSI_WINDOW)).iloc[-1]
    vols = calculate_volatility(trade_df, VOL_WINDOW).iloc[-1]
    prices = trade_df.iloc[-1]
    
    rank_data = []
    for name in ASSETS.values():
        if name in scores:
            rank_data.append({
                "名称": name,
                "综合得分": scores[name],
                "RSI(14)": rsis[name],
                "波动率": vols[name],
                "状态": "✅" if (prices[name]>mas[name] and scores[name]>0 and rsis[name]<RSI_LIMIT) else "❌"
            })
            
    rank_df = pd.DataFrame(rank_data).sort_values("综合得分", ascending=False).reset_index(drop=True)
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("💡 智能持仓建议")
        
        # 模拟选股
        candidates = rank_df[rank_df['状态']=="✅"].head(HOLD_COUNT)
        
        if candidates.empty:
            st.warning("🛑 **建议空仓**：市场风险过高 (RSI过热 或 趋势走坏)")
        else:
            # 计算建议权重
            cand_vols = candidates['波动率']
            inv_vols = 1 / (cand_vols + 0.0001)
            weights = inv_vols / inv_vols.sum()
            
            st.success("✅ **建议买入组合**")
            for name, w in weights.items():
                st.write(f"**{name}**: 仓位 **{w*100:.1f}%**")
            st.caption("注：仓位根据波动率动态分配，波动越小占比越大。")

    with c2:
        st.subheader("🔍 因子监控面板")
        display_df = rank_df.copy()
        display_df['综合得分'] = display_df['综合得分'].apply(lambda x: f"{x*100:.2f}%")
        display_df['RSI(14)'] = display_df['RSI(14)'].apply(lambda x: f"{x:.1f}")
        
        # 高亮 RSI 过热
        def highlight_rsi(val):
            v = float(val)
            return 'color: red' if v > RSI_LIMIT else ''
            
        st.dataframe(display_df.style.applymap(highlight_rsi, subset=['RSI(14)']), use_container_width=True)

    with st.expander("查看调仓历史 (含权重)"):
        h_df = pd.DataFrame({"日期": strategy_nav.index[-10:], "持仓详情": pos_history[-10:]}).sort_values("日期", ascending=False)
        st.table(h_df)
