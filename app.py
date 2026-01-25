import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 1. 页面配置与 URL 持久化 =================
st.set_page_config(page_title="全球动能工厂-全功能优化版", page_icon="🏭", layout="wide")

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

# ================= 4. 高速计算引擎 =================
@st.cache_data(ttl=3600)
def get_optimized_data(assets_keys, start_date):
    targets = {**st.session_state.my_assets, **BENCHMARKS}
    data = yf.download(list(targets.keys()), start=start_date, progress=False)
    df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
    df.index = df.index.tz_localize(None)
    return df.rename(columns=targets).ffill().dropna(how='all')

@st.cache_data
def run_full_backtest(df_all, rs, rl, rw, h, m):
    trade_names = [n for n in st.session_state.my_assets.values() if n in df_all.columns]
    df_t = df_all[trade_names]
    
    # 向量化计算
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    factor_results = [] # 用于分层分析

    score_vals = scores.values
    price_vals = df_t.values
    ma_vals = ma.values
    ret_vals = rets.values

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        mask = (s_row > 0) & (price_vals[i] > ma_vals[i])
        
        # 记录因子数据
        day_ranks = pd.Series(s_row).rank(ascending=False, method='first')
        for idx_asset in range(len(s_row)):
            factor_results.append({"Rank": int(day_ranks[idx_asset]), "Return": ret_vals[i+1][idx_asset]})

        day_pnl = 0.0
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_row[idx])[-h:]]
            day_pnl = np.mean(ret_vals[i+1][top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
    
    return pd.DataFrame({"nav": nav}, index=df_t.index).iloc[warm_up:], pd.DataFrame(factor_results), scores, ma, df_t

# ================= 5. 渲染界面 =================
st.title("🏭 全球动能工厂 (专业优化版)")
df = get_optimized_data(tuple(st.session_state.my_assets.keys()), start_d)

if not df.empty:
    nav_df, factor_df, score_df, ma_df, df_trade = run_full_backtest(df, rs, rl, rw, h, m)
    nav = nav_df['nav']
    
    # --- 今日信号 (置顶显示，辅助决策) ---
    st.divider()
    latest_scores = score_df.iloc[-1]
    latest_prices = df_trade.iloc[-1]
    latest_mas = ma_df.iloc[-1]
    
    rank_list = []
    for name in latest_scores.index:
        s, p, m_val = latest_scores[name], latest_prices[name], latest_mas[name]
        status = "✅ 持有" if (s > 0 and p > m_val) else "❌ 空仓"
        rank_list.append({"名称": name, "动能评分": s, "价格": p, "止损线": m_val, "信号": status})
    
    rank_df = pd.DataFrame(rank_list).sort_values("动能评分", ascending=False).reset_index(drop=True)
    
    c_sig1, c_sig2 = st.columns([1, 2])
    with c_sig1:
        st.subheader("📢 操作建议")
        buys = rank_df[rank_df['信号'] == "✅ 持有"].head(h)
        if buys.empty: st.error("🛑 信号：全额空仓")
        else:
            st.success(f"建议买入 Top {len(buys)}")
            for n in buys['名称']: st.write(f"- **{n}**")
            
    with c_sig2:
        st.subheader("📊 动能实时排行榜")
        st.dataframe(rank_df.style.format({"动能评分": "{:.2%}", "价格": "{:.3f}", "止损线": "{:.3f}"})
                     .applymap(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']),
                     use_container_width=True)

    # --- 策略图表 ---
    st.divider()
    fig = go.Figure()
    # 背景变色优化
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
            fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b, line=dict(dash='dot', color='gray')))
    
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- 绩效指标 ---
    days = (nav.index[-1] - nav.index[0]).days
    cagr = (nav.iloc[-1]**(365/days)-1)
    dr = nav.pct_change().dropna()
    sharpe = (dr.mean() * 252 - 0.02) / (dr.std() * np.sqrt(252))
    mdd = ((nav - nav.cummax())/nav.cummax()).min()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
    k2.metric("年化(CAGR)", f"{cagr:.2%}")
    k3.metric("夏普比率", f"{sharpe:.2f}")
    k4.metric("最大回撤", f"{mdd:.2%}")

    # --- 分层分析 (Factor Analysis) ---
    st.divider()
    st.subheader("🔬 因子体检：排名与收益关系")
    if not factor_df.empty:
        analysis = factor_df.groupby("Rank")["Return"].mean() * 100
        fig_bar = px.bar(x=analysis.index, y=analysis.values, title="不同排名位置的次日平均收益率",
                         labels={'x':'动能排名', 'y':'平均涨幅 (%)'}, color=analysis.values, color_continuous_scale="RdYlGn")
        fig_bar.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.error("数据加载失败，请检查网络环境。")
