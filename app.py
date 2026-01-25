import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-旗舰版", page_icon="🏭", layout="wide")

# 默认策略参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}

# --- 重要：初始化 Session State ---
# 确保在任何组件加载前，rs, rl 等变量已存在
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        # 优先从 URL 获取，没有则用默认值
        url_val = st.query_params.get(key, val)
        st.session_state[key] = int(url_val)

def update_url():
    """将当前参数同步到 URL，方便分享"""
    st.query_params.update({
        "rs": st.session_state.rs, 
        "rl": st.session_state.rl,
        "rw": st.session_state.rw, 
        "h": st.session_state.h, 
        "m": st.session_state.m
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
        nc = c1.text_input("代码", key="new_code_input", placeholder="AAPL")
        nn = c2.text_input("名称", key="new_name_input", placeholder="苹果")
        if st.button("➕ 添加标的"):
            if nc and nn: 
                st.session_state.my_assets[nc] = nn
                st.rerun()
    
    st.subheader("策略参数")
    # 使用 value=st.session_state.rs 确保双向绑定
    rs = st.slider("短期ROC (天)", 5, 60, value=st.session_state.rs, key="rs", on_change=update_url)
    rl = st.slider("长期ROC (天)", 30, 250, value=st.session_state.rl, key="rl", on_change=update_url)
    rw = st.slider("短期权重 (%)", 0, 100, value=st.session_state.rw, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, value=st.session_state.h, key="h", on_change=update_url)
    m = st.number_input("止损均线 (MA)", 5, 120, value=st.session_state.m, key="m", on_change=update_url)
    start_d = st.date_input("回测开始日期", datetime.date(2020, 1, 1))

# ================= 4. 高速数据引擎 =================
@st.cache_data(ttl=3600)
def get_optimized_data(assets_dict, start_date):
    targets = {**assets_dict, **BENCHMARKS}
    try:
        # 增加超时控制
        data = yf.download(list(targets.keys()), start=start_date, progress=False, timeout=30)
        if data.empty: return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        else:
            df = data
            
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except Exception as e:
        st.error(f"⚠️ 数据接口异常: {e}")
        return pd.DataFrame()

# ================= 5. 回测逻辑 =================
@st.cache_data
def run_full_backtest(df_all, rs, rl, rw, h, m):
    trade_names = [n for n in st.session_state.my_assets.values() if n in df_all.columns]
    df_t = df_all[trade_names]
    
    # 计算 ROC 和 均线
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    if len(df_t) <= warm_up: return None, None, scores, ma, df_t

    nav = np.ones(len(df_t))
    factor_results = [] 
    score_vals, price_vals, ma_vals, ret_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        if not np.isnan(s_row).all():
            day_ranks = pd.Series(s_row).rank(ascending=False, method='first')
            for idx_asset in range(len(s_row)):
                r_val, ret_val = day_ranks.iloc[idx_asset], ret_vals[i+1][idx_asset]
                if pd.notnull(r_val) and pd.notnull(ret_val):
                    factor_results.append({"Rank": int(r_val), "Return": ret_val})

        mask = (s_row > 0) & (price_vals[i] > ma_vals[i])
        day_pnl = 0.0
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_row[idx])[-h:]]
            day_pnl = np.nanmean(ret_vals[i+1][top_idx])
        nav[i+1] = nav[i] * (1 + day_pnl)
    
    return pd.DataFrame({"nav": nav}, index=df_t.index).iloc[warm_up:], pd.DataFrame(factor_results), scores, ma, df_t

# ================= 6. UI 渲染 =================
st.title("🏭 全球动能工厂")
st.info("🚀 状态：2026 修复版已成功初始化参数环境。")

df = get_optimized_data(st.session_state.my_assets, start_d)

if not df.empty:
    nav_df, factor_df, score_df, ma_df, df_trade = run_full_backtest(df, rs, rl, rw, h, m)
    
    if nav_df is not None:
        nav = nav_df['nav']
        
        # --- 实时排行 ---
        st.divider()
        st.subheader("📊 今日实时信号")
        latest_scores, latest_prices, latest_mas = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
        rank_list = []
        for name in latest_scores.index:
            s, p, mv = latest_scores[name], latest_prices[name], latest_mas[name]
            status = "✅ 持有" if (s > 0 and p > mv) else "❌ 空仓"
            rank_list.append({"名称": name, "评分": s, "价格": p, "止损线": mv, "信号": status})
        
        rank_df = pd.DataFrame(rank_list).sort_values("评分", ascending=False)
        st.dataframe(rank_df.style.format({"评分": "{:.2%}", "价格": "{:.2f}"})
                     .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']),
                     width="stretch")

        # --- 回测图表 ---
        st.divider()
        st.subheader("📈 策略净值走势")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name="动能策略", line=dict(color='#00ff88', width=3)))
        for b_name in BENCHMARKS.values():
            if b_name in df.columns:
                b_nav = df[b_name].loc[nav.index[0]:]
                b_nav = b_nav / b_nav.iloc[0]
                fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b_name, line=dict(dash='dot')))
        
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        # --- 指标卡 ---
        k1, k2, k3 = st.columns(3)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("年化收益", f"{(nav.iloc[-1]**(365/(nav.index[-1]-nav.index[0]).days)-1):.2%}")
        k3.metric("最大回撤", f"{((nav - nav.cummax())/nav.cummax()).min():.2%}")
        
    else:
        st.warning("⚠️ 数据量不足以计算 ROC，请增加回测日期范围。")
else:
    st.error("📡 正在获取数据... 若长时间不显示，请检查 requirements.txt 是否已更新并重启。")
