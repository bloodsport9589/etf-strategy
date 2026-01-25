import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026避险版", page_icon="🛡️", layout="wide")

# 默认参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 80, "h": 1, "m": 20}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = int(st.query_params.get(key, val))

# 混合品种池：权益 + 防御 + 现金(模拟)
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", 
    "511130.SS": "30年国债ETF", 
    "518880.SS": "黄金ETF",
    "510300.SS": "沪深300",
    "588050.SS": "科创50",
    "511880.SS": "银华日利" # 模拟现金/货币基金
}
BENCHMARKS = {"510300.SS": "沪深300"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 2. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    with st.expander("📝 品种池管理"):
        c1, c2 = st.columns([2, 1])
        nc, nn = c1.text_input("代码"), c2.text_input("名称")
        if st.button("➕ 添加"):
            if nc and nn: 
                st.session_state.my_assets[nc] = nn
                st.rerun()
        if st.button("🔄 恢复默认"):
            st.session_state.my_assets = DEFAULT_ASSETS.copy()
            st.rerun()

    st.divider()
    rs = st.slider("短期周期", 5, 60, value=st.session_state.rs, key="rs")
    rl = st.slider("长期周期", 30, 250, value=st.session_state.rl, key="rl")
    rw = st.slider("短期权重 %", 0, 100, value=st.session_state.rw, key="rw") / 100.0
    h = st.number_input("持仓数", 1, 5, value=st.session_state.h, key="h")
    m = st.number_input("风控均线", 5, 120, value=st.session_state.m, key="m")
    start_d = st.date_input("回测开始", datetime.date(2021, 1, 1))

# ================= 3. 高效引擎 =================
@st.cache_data(ttl=3600)
def get_data(assets_dict, start_date):
    targets = {**assets_dict, **BENCHMARKS}
    try:
        data = yf.download(list(targets.keys()), start=start_date, progress=False)
        df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except: return pd.DataFrame()

@st.cache_data
def run_backtest(df_all, assets, rs, rl, rw, h, m):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    df_t = df_all[trade_names]
    # 计算评分
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    holdings = [[] for _ in range(len(df_t))]
    
    s_vals, p_vals, m_vals, r_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        # 绝对动能过滤：只有评分 > 0 且 价格 > 均线 才考虑买入
        mask = (s_vals[i] > 0) & (p_vals[i] > m_vals[i])
        day_pnl = 0.0
        curr_h = []
        
        if np.any(mask):
            idx = np.where(mask)[0]
            # 选出动能最强的 h 个
            top_idx = idx[np.argsort(s_vals[i][idx])[-h:]]
            day_pnl = np.nanmean(r_vals[i+1][top_idx])
            curr_h = [trade_names[j] for j in top_idx]
        
        # 如果没有任何品种满足条件，自动进入“空仓/现金状态”（收益率为 0）
        nav[i+1] = nav[i] * (1 + day_pnl)
        holdings[i+1] = curr_h
            
    return pd.DataFrame({"nav": nav, "h": holdings}, index=df_t.index).iloc[warm_up:]

# ================= 4. 智能 UI =================
st.title("🛡️ 全球动能工厂 (绝对动能增强版)")
st.markdown("> **观察点**：当所有品种都不符合“强趋势”时，策略会主动选择**空仓**。你会发现回撤曲线在此时会变成一条直线。")

df = get_data(st.session_state.my_assets, start_d)

if not df.empty:
    res = run_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m)
    nav = res['nav']
    
    # KPI 
    mdd = ((nav - nav.cummax()) / nav.cummax()).min()
    daily_ret = nav.pct_change().dropna()
    sharpe = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252)) if not daily_ret.empty else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
    c2.metric("最大回撤", f"{mdd:.2%}", delta="越小越安全")
    c3.metric("夏普比率", f"{sharpe:.2f}")
    c4.metric("年化收益", f"{(nav.iloc[-1]**(365/(nav.index[-1]-nav.index[0]).days)-1):.2%}")

    # 绘图
    st.divider()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="动能策略(含空仓过滤)", line=dict(color='#00ff88', width=3)))
    
    # 对比基准
    if "沪深300" in df.columns:
        b_nav = df["沪深300"].loc[nav.index[0]:]
        fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name="沪深300 (基准)", line=dict(dash='dot', color='gray')))

    fig.update_layout(template="plotly_dark", height=500, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    # 底部持仓检查
    st.divider()
    latest = res['h'].iloc[-1]
    if latest:
        st.success(f"🚀 当前建议持仓：{' | '.join(latest)}")
    else:
        st.warning("💤 策略信号：目前无强势标的，建议【全额空仓】避险。")
