import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 1. 页面配置与 URL 持久化 =================
st.set_page_config(page_title="全球动能工厂-旗舰版", page_icon="🏭", layout="wide")

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
    
    with st.expander("📝 品种管理", expanded=False):
        c1, c2 = st.columns(2)
        nc = c1.text_input("代码", key="new_code", placeholder="AAPL")
        nn = c2.text_input("名称", key="new_name", placeholder="苹果")
        if st.button("➕ 添加标的"):
            if nc and nn: 
                st.session_state.my_assets[nc] = nn
                st.rerun()
    
    st.subheader("策略参数")
    rs = st.slider("短期ROC (天)", 5, 60, int(query_params.get("rs", DEFAULTS["rs"])), key="rs", on_change=update_url)
    rl = st.slider("长期ROC (天)", 30, 250, int(query_params.get("rl", DEFAULTS["rl"])), key="rl", on_change=update_url)
    rw = st.slider("短期权重 (%)", 0, 100, int(query_params.get("rw", DEFAULTS["rw"])), key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, int(query_params.get("h", DEFAULTS["h"])), key="h", on_change=update_url)
    m = st.number_input("止损均线 (MA)", 5, 120, int(query_params.get("m", DEFAULTS["m"])), key="m", on_change=update_url)
    start_d = st.date_input("回测开始日期", datetime.date(2020, 1, 1))

# ================= 4. 高速数据引擎 =================
@st.cache_data(ttl=3600)
def get_optimized_data(assets_keys, start_date):
    targets = {**st.session_state.my_assets, **BENCHMARKS}
    data = yf.download(list(targets.keys()), start=start_date, progress=False, timeout=20)
    if data.empty: return pd.DataFrame()
    df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
    df.index = df.index.tz_localize(None)
    return df.rename(columns=targets).ffill().dropna(how='all')

# ================= 5. 回测与因子统计 =================
@st.cache_data
def run_full_backtest(df_all, rs, rl, rw, h, m):
    trade_names = [n for n in st.session_state.my_assets.values() if n in df_all.columns]
    df_t = df_all[trade_names]
    
    # 评分计算
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    factor_results = [] 

    score_vals = scores.values
    price_vals = df_t.values
    ma_vals = ma.values
    ret_vals = rets.values

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        
        # --- 因子体检记录 ---
        if not np.isnan(s_row).all():
            day_ranks = pd.Series(s_row).rank(ascending=False, method='first')
            for idx_asset in range(len(s_row)):
                r_val = day_ranks.iloc[idx_asset]
                ret_val = ret_vals[i+1][idx_asset]
                if pd.notnull(r_val) and pd.notnull(ret_val):
                    factor_results.append({"Rank": int(r_val), "Return": ret_val})

        # --- 策略模拟 ---
        mask = (s_row > 0) & (price_vals[i] > ma_vals[i])
        day_pnl = 0.0
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_row[idx])[-h:]]
            day_pnl = np.nanmean(ret_vals[i+1][top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
    
    return pd.DataFrame({"nav": nav}, index=df_t.index).iloc[warm_up:], pd.DataFrame(factor_results), scores, ma, df_t

# ================= 6. UI 界面渲染 =================
st.title("🏭 全球动能工厂 (旗舰优化版)")
df = get_optimized_data(tuple(st.session_state.my_assets.keys()), start_d)

if not df.empty:
    nav_df, factor_df, score_df, ma_df, df_trade = run_full_backtest(df, rs, rl, rw, h, m)
    nav = nav_df['nav']
    
    # --- Part 1: 今日实时信号 ---
    st.divider()
    latest_scores = score_df.iloc[-1]
    latest_prices = df_trade.iloc[-1]
    latest_mas = ma_df.iloc[-1]
    
    rank_list = []
    for name in latest_scores.index:
        s, p, mv = latest_scores[name], latest_prices[name], latest_mas[name]
        status = "✅ 持有" if (s > 0 and p > mv) else "❌ 空仓"
        rank_list.append({"名称": name, "动能评分": s, "价格": p, "止损线": mv, "信号": status})
    
    rank_df = pd.DataFrame(rank_list).sort_values("动能评分", ascending=False).reset_index(drop=True)
    
    c_s1, c_s2 = st.columns([1, 2])
    with c_s1:
        st.subheader("📢 今日操作建议")
        buys = rank_df[rank_df['信号'] == "✅ 持有"].head(h)
        if buys.empty: st.error("🛑 策略建议：全额空仓/持有现金")
        else:
            st.success(f"建议持有前 {len(buys)} 个品种:")
            for n in buys['名称']: st.write(f"- **{n}**")
            
    with c_s2:
        st.subheader("📊 实时排行 (今日实时数据)")
        # 修复点 1：applymap -> map
        # 修复点 2：use_container_width=True -> width="stretch"
        st.dataframe(rank_df.style.format({"动能评分": "{:.2%}", "价格": "{:.3f}", "止损线": "{:.3f}"})
                     .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']),
                     width="stretch")

    # --- Part 2: 策略表现图表 ---
    st.divider()
    fig = go.Figure()
    # 趋势背景变色
    diff = (nav.diff() >= 0).astype(int)
    cp = diff.diff().fillna(0) != 0
    cp_idx = np.concatenate(([0], np.where(cp)[0], [len(nav)-1]))
    for j in range(len(cp_idx)-1):
        s, e = cp_idx[j], cp_idx[j+1]
        cl = "rgba(0, 255, 136, 0.08)" if diff.iloc[e] == 1 else "rgba(255, 68, 68, 0.08)"
        fig.add_vrect(x0=nav.index[s], x1=nav.index[e], fillcolor=cl, line_width=0, layer="below")

    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略净值", line=dict(color='#00ff88', width=3)))
    for b in BENCHMARKS.values():
        if b in df.columns:
            b_nav = df[b].loc[nav.index[0]:]; b_nav /= b_nav.iloc[0]
            fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b, line=dict(dash='dot', color='gray'))))
    
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
    # 修复点 3：use_container_width=True -> width="stretch"
    st.plotly_chart(fig, width="stretch")

    # --- Part 3: KPI 绩效面板 ---
    d_count = (nav.index[-1] - nav.index[0]).days
    cagr = (nav.iloc[-1]**(365/max(d_count,1))-1)
    daily_rets = nav.pct_change().dropna()
    sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252)) if len(daily_rets)>0 else 0
    mdd = ((nav - nav.cummax())/nav.cummax()).min()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("累计收益率", f"{nav.iloc[-1]-1:.2%}")
    k2.metric("年化收益率", f"{cagr:.2%}")
    k3.metric("夏普比率", f"{sharpe:.2f}")
    k4.metric("最大回撤", f"{mdd:.2%}")

    # --- Part 4: 因子有效性体检 ---
    st.divider()
    st.subheader("🔬 因子体检：回测期间排名与次日收益")
    if not factor_df.empty:
        analysis = factor_df.groupby("Rank")["Return"].mean() * 100
        fig_bar = px.bar(x=analysis.index, y=analysis.values, title="历史平均：各排名位置的次日表现",
                         labels={'x':'动能排名', 'y':'平均涨跌 (%)'}, color=analysis.values, color_continuous_scale="RdYlGn")
        fig_bar.update_layout(template="plotly_dark", height=400)
        # 修复点 4：use_container_width=True -> width="stretch"
        st.plotly_chart(fig_bar, width="stretch")
else:
    st.warning("📡 正在尝试连接全球数据服务器，请稍候。若长时间无响应请确认 GitHub 上的 requirements.txt 包含 yfinance。")
