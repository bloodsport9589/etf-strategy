import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 1. 页面配置 =================
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
def get_optimized_data(assets_dict, start_date):
    targets = {**assets_dict, **BENCHMARKS}
    try:
        # 增加超时控制，防止卡死
        data = yf.download(list(targets.keys()), start=start_date, progress=False, timeout=30)
        if data.empty:
            return pd.DataFrame()
        
        # 处理多级索引
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        else:
            df = data
            
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except Exception as e:
        st.error(f"数据抓取失败: {str(e)}")
        return pd.DataFrame()

# ================= 5. 回测引擎 =================
@st.cache_data
def run_full_backtest(df_all, rs, rl, rw, h, m):
    trade_names = [n for n in st.session_state.my_assets.values() if n in df_all.columns]
    df_t = df_all[trade_names]
    
    # 核心因子计算
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    if len(df_t) <= warm_up:
        return None, None, scores, ma, df_t

    nav = np.ones(len(df_t))
    factor_results = [] 

    score_vals = scores.values
    price_vals = df_t.values
    ma_vals = ma.values
    ret_vals = rets.values

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        
        # 记录因子排名表现
        if not np.isnan(s_row).all():
            day_ranks = pd.Series(s_row).rank(ascending=False, method='first')
            for idx_asset in range(len(s_row)):
                r_val = day_ranks.iloc[idx_asset]
                ret_val = ret_vals[i+1][idx_asset]
                if pd.notnull(r_val) and pd.notnull(ret_val):
                    factor_results.append({"Rank": int(r_val), "Return": ret_val})

        # 策略买入逻辑：动能>0 且 价格>均线
        mask = (s_row > 0) & (price_vals[i] > ma_vals[i])
        day_pnl = 0.0
        if np.any(mask):
            idx = np.where(mask)[0]
            # 选分值最高的 h 个
            top_idx = idx[np.argsort(s_row[idx])[-h:]]
            day_pnl = np.nanmean(ret_vals[i+1][top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
    
    nav_series = pd.DataFrame({"nav": nav}, index=df_t.index).iloc[warm_up:]
    return nav_series, pd.DataFrame(factor_results), scores, ma, df_t

# ================= 6. UI 渲染逻辑 =================
st.title("🏭 全球动能工厂 (2026 旗舰版)")

# 获取数据
df = get_optimized_data(st.session_state.my_assets, start_d)

if df.empty:
    st.error("❌ 无法获取行情数据。请检查网络连接，或确认标的代码（如纳指ETF 513100.SS）是否正确。")
else:
    nav_df, factor_df, score_df, ma_df, df_trade = run_full_backtest(df, rs, rl, rw, h, m)
    
    if nav_df is None:
        st.warning(f"⚠️ 数据量不足以进行回测。当前预热期需要 {max(rs, rl, m)} 天数据，请调整开始日期。")
    else:
        nav = nav_df['nav']
        
        # --- Part 1: 实时信号台 ---
        st.divider()
        st.subheader("📢 实时交易雷达")
        
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
            buys = rank_df[rank_df['信号'] == "✅ 持有"].head(h)
            if buys.empty: 
                st.error("🚨 策略建议：全额空仓")
            else:
                st.success(f"建议持仓：{', '.join(buys['名称'].tolist())}")
                st.info("配比建议：等权分配持仓权重")
                
        with c_s2:
            # 适配 2026 语法: map 替代 applymap, width="stretch" 替代 use_container_width
            st.dataframe(rank_df.style.format({"动能评分": "{:.2%}", "价格": "{:.3f}", "止损线": "{:.3f}"})
                         .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']),
                         width="stretch")

        # --- Part 2: 净值曲线图 ---
        st.divider()
        st.subheader("📈 策略净值走势")
        
        fig = go.Figure()
        # 绘制背景色块（持有期/空仓期）
        diff = (nav.diff() >= 0).astype(int)
        cp = diff.diff().fillna(0) != 0
        cp_idx = np.concatenate(([0], np.where(cp)[0], [len(nav)-1]))
        for j in range(len(cp_idx)-1):
            s, e = cp_idx[j], cp_idx[j+1]
            cl = "rgba(0, 255, 136, 0.05)" if diff.iloc[e] == 1 else "rgba(255, 68, 68, 0.05)"
            fig.add_vrect(x0=nav.index[s], x1=nav.index[e], fillcolor=cl, line_width=0, layer="below")

        # 策略主线
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name="动能策略", line=dict(color='#00ff88', width=3)))
        
        # 基准线
        for b_name in BENCHMARKS.values():
            if b_name in df.columns:
                b_nav = df[b_name].loc[nav.index[0]:]
                b_nav = b_nav / b_nav.iloc[0]
                fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b_name, line=dict(dash='dot', color='gray')))
        
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        # --- Part 3: 核心绩效指标 ---
        c_days = (nav.index[-1] - nav.index[0]).days
        cagr = (nav.iloc[-1]**(365/max(c_days, 1)) - 1)
        daily_rets = nav.pct_change().dropna()
        sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252)) if len(daily_rets)>0 else 0
        mdd = ((nav - nav.cummax())/nav.cummax()).min()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("年化收益 (CAGR)", f"{cagr:.2%}")
        k3.metric("夏普比率", f"{sharpe:.2f}")
        k4.metric("最大回撤", f"{mdd:.2%}")

        # --- Part 4: 因子体检 ---
        if not factor_df.empty:
            st.divider()
            st.subheader("🔬 动能因子有效性验证")
            analysis = factor_df.groupby("Rank")["Return"].mean() * 100
            fig_bar = px.bar(x=analysis.index, y=analysis.values, 
                             title="各动能排名次日的平均涨跌幅 (%)",
                             labels={'x':'动能排名 (1为最高)', 'y':'次日收益率 (%)'}, 
                             color=analysis.values, color_continuous_scale="RdYlGn")
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, width="stretch")

# 底部页脚
st.caption("注：本工具仅供回测研究，不构成投资建议。数据来源：Yahoo Finance。")
