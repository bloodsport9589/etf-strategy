import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 页面配置与 URL 持久化 =================
st.set_page_config(page_title="全球动能工厂-极速版", page_icon="⚡", layout="wide")

# 默认参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
query_params = st.query_params

def update_url():
    st.query_params.update({
        "rs": st.session_state.rs, "rl": st.session_state.rl,
        "rw": st.session_state.rw, "h": st.session_state.h, "m": st.session_state.m
    })

# ================= 2. 标的池配置 =================
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 3. 侧边栏控制 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    
    # 标的管理 (简化以提速)
    with st.expander("📝 品种管理"):
        c1, c2 = st.columns(2)
        nc = c1.text_input("代码", key="new_code")
        nn = c2.text_input("名称", key="new_name")
        if st.button("添加"):
            if nc and nn: 
                st.session_state.my_assets[nc] = nn
                st.rerun()
    
    st.subheader("策略参数")
    rs = st.slider("短期ROC", 5, 60, int(query_params.get("rs", DEFAULTS["rs"])), key="rs", on_change=update_url)
    rl = st.slider("长期ROC", 30, 250, int(query_params.get("rl", DEFAULTS["rl"])), key="rl", on_change=update_url)
    rw = st.slider("短期权重%", 0, 100, int(query_params.get("rw", DEFAULTS["rw"])), key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数", 1, 10, int(query_params.get("h", DEFAULTS["h"])), key="h", on_change=update_url)
    m = st.number_input("止损均线", 5, 120, int(query_params.get("m", DEFAULTS["m"])), key="m", on_change=update_url)
    start_d = st.date_input("开始日期", datetime.date(2020, 1, 1))

# ================= 4. 高速数据引擎 =================
@st.cache_data(ttl=3600)
def get_optimized_data(assets_keys, start_date):
    targets = {**st.session_state.my_assets, **BENCHMARKS}
    # 批量下载是提速关键
    data = yf.download(list(targets.keys()), start=start_date, progress=False)
    df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
    df.index = df.index.tz_localize(None)
    return df.rename(columns=targets).ffill().dropna(how='all')

# ================= 5. 高速回测引擎 =================
@st.cache_data
def run_fast_backtest(df_all, rs, rl, rw, h, m):
    trade_names = [n for n in st.session_state.my_assets.values() if n in df_all.columns]
    df_t = df_all[trade_names]
    
    # 向量化计算指标
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    holdings = ["空仓"] * len(df_t)
    
    # 核心循环提速：减少对象创建
    score_vals = scores.values
    price_vals = df_t.values
    ma_vals = ma.values
    ret_vals = rets.values
    names = np.array(trade_names)

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        # 筛选逻辑优化
        mask = (s_row > 0) & (price_vals[i] > ma_vals[i])
        day_pnl = 0.0
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_row[idx])[-h:]]
            day_pnl = np.mean(ret_vals[i+1][top_idx])
            holdings[i+1] = "<br>".join([f"{names[j]}: {price_vals[i+1][j]:.2f}" for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
    
    return pd.DataFrame({"nav": nav, "holdings": holdings}, index=df_t.index).iloc[warm_up:]

# ================= 6. 界面渲染 =================
st.title("⚡ 全球动能工厂 (极速优化版)")
df = get_optimized_data(tuple(st.session_state.my_assets.keys()), start_d)

if not df.empty:
    res = run_fast_backtest(df, rs, rl, rw, h, m)
    nav = res['nav']
    
    # KPI 计算
    days = (nav.index[-1] - nav.index[0]).days
    cagr = (nav.iloc[-1]**(365/days)-1)
    daily_rets = nav.pct_change().dropna()
    sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252))
    mdd = ((nav - nav.cummax())/nav.cummax()).min()

    # 绘图优化：区间变色而非每日变色
    fig = go.Figure()
    
    # 1. 优化背景色块生成 (合并连续趋势以减少对象数)
    diff = (nav.diff() >= 0).astype(int)
    change_points = diff.diff().fillna(0) != 0
    cp_idx = np.where(change_points)[0]
    cp_idx = np.concatenate(([0], cp_idx, [len(nav)-1]))
    
    for j in range(len(cp_idx)-1):
        start, end = cp_idx[j], cp_idx[j+1]
        color = "rgba(0, 255, 136, 0.08)" if diff.iloc[end] == 1 else "rgba(255, 68, 68, 0.08)"
        fig.add_vrect(x0=nav.index[start], x1=nav.index[end], fillcolor=color, line_width=0, layer="below")

    # 2. 绘图
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略", line=dict(color='#00ff88', width=2.5),
                             customdata=res['holdings'], hovertemplate="%{x}<br>净值: %{y:.3f}<br>%{customdata}<extra></extra>"))
    
    # 3. 基准
    for b in BENCHMARKS.values():
        if b in df.columns:
            b_nav = df[b].loc[nav.index[0]:]; b_nav /= b_nav.iloc[0]
            fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b, line=dict(dash='dot', color='gray')))

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
    c2.metric("年化(CAGR)", f"{cagr:.2%}")
    c3.metric("夏普比率", f"{sharpe:.2f}")
    c4.metric("最大回撤", f"{mdd:.2%}")
    
    # 今日信号 (单独渲染以提速)
    st.divider()
    st.subheader("📢 今日实时信号")
    # ... 此处逻辑保持精简 ...
