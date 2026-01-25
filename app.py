import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# ================= 1. 基础配置与状态管理 =================
st.set_page_config(page_title="全球动能工厂-2026旗舰版", page_icon="🏭", layout="wide")

# 核心参数默认值
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}

# --- 初始化参数 ---
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        try:
            p_val = st.query_params.get(key, str(val))
            st.session_state[key] = int(p_val)
        except:
            st.session_state[key] = val

# 标的池初始化
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# --- URL 更新函数 ---
def update_url():
    params = {k: st.session_state[k] for k in DEFAULTS.keys() if k in st.session_state}
    st.query_params.update(params)

# ================= 2. 侧边栏设置 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    
    # --- A. 回测区间选择 ---
    with st.expander("📅 回测区间设置", expanded=True):
        col_d1, col_d2 = st.columns(2)
        default_start = datetime.date.today() - datetime.timedelta(days=365*3)
        default_end = datetime.date.today()
        
        start_d = col_d1.date_input("开始日期", default_start)
        end_d = col_d2.date_input("结束日期", default_end)
        
        if start_d >= end_d:
            st.error("⚠️ 结束日期必须晚于开始日期")

    # --- B. 品种管理 ---
    with st.expander("📝 品种管理", expanded=False):
        st.markdown("**添加/删除标的**")
        c1, c2 = st.columns([2, 1])
        nc = c1.text_input("代码", key="input_code")
        nn = c2.text_input("名称", key="input_name")
        if st.button("➕ 添加", width="stretch"):
            if nc and nn:
                st.session_state.my_assets[nc] = nn
                st.rerun()
        
        st.divider()
        for code, name in list(st.session_state.my_assets.items()):
            cols = st.columns([3, 1])
            cols[0].write(f"{name} ({code})")
            if cols[1].button("❌", key=f"del_{code}"):
                del st.session_state.my_assets[code]
                st.rerun()
    
    st.divider()
    rs = st.slider("短期评分周期 (天)", 5, 60, key="rs", on_change=update_url)
    rl = st.slider("长期评分周期 (天)", 30, 250, key="rl", on_change=update_url)
    rw = st.slider("权重分配 (短期%)", 0, 100, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, key="h", on_change=update_url)
    m = st.number_input("风控均线 (MA)", 5, 120, key="m", on_change=update_url)

# ================= 3. 数据引擎 (纯净版 - 无UI副作用) =================
@st.cache_data(ttl=3600)
def get_clean_data(assets_dict, start_date, end_date):
    """
    注意：此函数内部不再包含 st.toast 或 st.error，以避免缓存重放错误。
    所有 UI 提示都移动到主流程中。
    """
    targets = {**assets_dict, **BENCHMARKS}
    
    fetch_start = start_date - timedelta(days=365)
    fetch_end = end_date + timedelta(days=1)
    
    try:
        data = yf.download(list(targets.keys()), start=fetch_start, end=fetch_end, progress=False)
        
        if data.empty:
            return pd.DataFrame()

        # 兼容性处理
        if 'Adj Close' in data.columns:
             df = data['Adj Close']
        elif isinstance(data.columns, pd.MultiIndex) and 'Adj Close' in data.columns.levels[0]:
             df = data['Adj Close']
        elif 'Close' in data.columns:
             df = data['Close']
        else:
             df = data 

        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        valid_cols = set(df.columns)
        rename_map = {k: v for k, v in targets.items() if k in valid_cols}
        df = df.rename(columns=rename_map)
        
        df.index = df.index.tz_localize(None)
        df = df.ffill().dropna(how='all')
        
        return df

    except Exception:
        return pd.DataFrame()

@st.cache_data
def run_enhanced_backtest(df_all, assets, rs, rl, rw, h, m, user_start_date):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None, 0
    
    df_t = df_all[trade_names]
    
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    trade_count = 0

    s_vals, p_vals, m_vals, r_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        mask = (s_vals[i] > 0) & (p_vals[i] > m_vals[i])
        day_pnl = 0.0
        curr_h = []
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_vals[i][idx])[-h:]]
            day_pnl = np.nanmean(r_vals[i+1][top_idx])
            curr_h = sorted([trade_names[j] for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
        if hist[i+1] != hist[i]: trade_count += 1
            
    full_res = pd.DataFrame({"nav": nav, "holdings": hist}, index=df_t.index)
    
    mask_slice = full_res.index >= pd.to_datetime(user_start_date)
    res_sliced = full_res.loc[mask_slice].copy()
    
    if res_sliced.empty:
        return None, None, None, None, 0
        
    res_sliced['nav'] = res_sliced['nav'] / res_sliced['nav'].iloc[0]
    
    scores_sliced = scores.loc[mask_slice]
    ma_sliced = ma.loc[mask_slice]
    df_t_sliced = df_t.loc[mask_slice]
    
    return res_sliced, scores_sliced, ma_sliced, df_t_sliced, trade_count

# ================= 4. UI 渲染 =================
st.title("🏭 全球动能工厂")

# --- UI 提示移动到这里 ---
st.toast(f"正在更新数据: {start_d} ~ {end_d}", icon="📡")

# 调用纯净版数据函数
df = get_clean_data(st.session_state.my_assets, start_d, end_d)

if not df.empty:
    bt = run_enhanced_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m, start_d)
    res_df, score_df, ma_df, df_trade, t_count = bt if bt[0] is not None else (None, None, None, None, 0)
    
    if res_df is not None:
        nav = res_df['nav']
        
        # 指标卡
        mdd = ((nav - nav.cummax()) / nav.cummax()).min()
        daily_rets = nav.pct_change().dropna()
        days_period = (nav.index[-1] - nav.index[0]).days
        ann_factor = 365 / max(days_period, 1)
        sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252)) if not daily_rets.empty else 0
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("区间累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("年化收益", f"{(nav.iloc[-1]**ann_factor - 1):.2%}")
        k3.metric("区间最大回撤", f"{mdd:.2%}", delta_color="inverse")
        k4.metric("夏普比率", f"{sharpe:.2f}")
        k5.metric("调仓/交易日", f"{t_count} 次 / {len(nav)} 天")

        # 绘图
        st.divider()
        st.subheader(f"📈 策略净值走势 ({start_d} 至 {end_d})")
        
        fig = go.Figure()

        ma_line = nav.rolling(10).mean()
        status = (nav >= ma_line).astype(int)
        change_idx = np.where(status.diff().fillna(0) != 0)[0]
        segs = np.concatenate(([0], change_idx, [len(nav)-1]))
        for i in range(len(segs)-1):
            cl = "rgba(0, 255, 136, 0.06)" if status.iloc[segs[i+1]] == 1 else "rgba(255, 68, 68, 0.06)"
            fig.add_vrect(x0=nav.index[segs[i]], x1=nav.index[segs[i+1]], fillcolor=cl, line_width=0, layer="below")

        fig.add_trace(go.Scatter(
            x=nav.index, y=nav, name="动能策略", 
            line=dict(color='#00ff88', width=3),
            text=[f"持仓: {', '.join(h) if h else '空仓'}" for h in res_df['holdings']],
            hoverinfo="x+y+text"
        ))

        re_dates = [res_df.index[i] for i in range(1, len(res_df)) if res_df['holdings'].iloc[i] != res_df['holdings'].iloc[i-1]]
        fig.add_trace(go.Scatter(
            x=re_dates, y=nav.loc[re_dates], mode='markers', name="调仓动作",
            marker=dict(symbol='diamond', size=6, color='white', line=dict(width=1, color='#00ff88')),
            hoverinfo="skip"
        ))

        for b_name in BENCHMARKS.values():
            if b_name in df.columns:
                b_nav = df[b_name].loc[nav.index]
                if not b_nav.empty:
                    b_nav = b_nav / b_nav.iloc[0]
                    fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b_name, 
                                             line=dict(dash='dot', width=1.2), opacity=0.6))

        fig.update_layout(
            template="plotly_dark", height=600, 
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width="stretch")

        # 榜单
        st.divider()
        st.subheader("📋 最新信号明细")
        if not score_df.empty:
            l_scores, l_prices, l_mas = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
            ranks = []
            for name in l_scores.index:
                price_ok = l_prices[name] > l_mas[name]
                score_ok = l_scores[name] > 0
                sig = "✅ 持有" if (score_ok and price_ok) else "❌ 空仓"
                
                ranks.append({
                    "名称": name, 
                    "动能评分": l_scores[name], 
                    "当前价格": l_prices[name], 
                    "均线止损": l_mas[name], 
                    "状态": "价格在线上" if price_ok else "价格在线下",
                    "信号": sig
                })
            
            df_rank = pd.DataFrame(ranks).sort_values("动能评分", ascending=False)
            
            st.dataframe(
                df_rank.style.format({"动能评分": "{:.2%}", "当前价格": "{:.3f}", "均线止损": "{:.3f}"})
                .map(lambda x: 'color: #00ff88; font-weight: bold' if "✅" in str(x) else 'color: #ff4444', subset=['信号'])
                .map(lambda x: 'color: #ffaa00' if x < 0 else 'color: #eeeeee', subset=['动能评分']),
                width="stretch"
            )
    else:
        st.warning("⚠️ 所选时间段内数据不足，无法回测。")
else:
    # --- 错误提示也在主流程处理 ---
    st.error("📡 数据获取失败，请检查网络或稍后重试。")
