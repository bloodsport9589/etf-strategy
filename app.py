import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026旗舰版", page_icon="🏭", layout="wide")

DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = DEFAULTS[key]

def update_url():
    params = {k: str(st.session_state[k]) for k in DEFAULTS.keys() if k in st.session_state}
    st.query_params.update(params)

# 标的池
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "511130.SS": "30年国债ETF", "510300.SS": "沪深300"
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 2. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    with st.expander("📝 品种管理", expanded=True):
        st.info("💡 A股后缀提示：上交所 .SS | 深交所 .SZ")
        c1, c2 = st.columns([2, 1])
        nc, nn = c1.text_input("代码", key="inc", placeholder="510300.SS"), c2.text_input("名称", key="inn", placeholder="沪深300")
        if st.button("➕ 确认添加", width="stretch"):
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
        if st.button("🔄 重置品种池"):
            st.session_state.my_assets = DEFAULT_ASSETS.copy()
            st.rerun()

    st.divider()
    st.slider("短期周期", 5, 60, key="rs", on_change=update_url)
    st.slider("长期周期", 30, 250, key="rl", on_change=update_url)
    st.slider("权重 %", 0, 100, key="rw", on_change=update_url)
    st.number_input("持仓数量", 1, 10, key="h", on_change=update_url)
    st.number_input("均线止损", 5, 120, key="m", on_change=update_url)
    
    st.divider()
    today = datetime.date.today()
    dr = st.date_input("回测区间", value=(today - datetime.timedelta(days=365), today), key="dr")

# ================= 3. 增强数据引擎 (Debug Mode) =================
@st.cache_data(ttl=3600)
def fetch_data_with_debug(assets_dict, start_date, end_date, warm_up):
    actual_start = start_date - datetime.timedelta(days=int(warm_up * 1.6) + 30)
    targets = {**assets_dict, **BENCHMARKS}
    tickers = list(targets.keys())
    
    with st.status("🚀 正在同步行情数据...", expanded=False) as status:
        try:
            # 2026 修复：增加 proxy 或 strings 设置以应对 API 限制
            data = yf.download(tickers, start=actual_start, end=end_date, progress=False, timeout=20)
            
            if data.empty:
                status.update(label="❌ 错误：Yahoo Finance 未返回任何数据", state="error")
                return pd.DataFrame()

            # 检查是否有品种完全没抓到数据
            missing = [t for t in tickers if data['Close'][t].isnull().all()] if isinstance(data.columns, pd.MultiIndex) else []
            if missing:
                st.warning(f"⚠️ 以下品种未获得数据，请检查代码：{', '.join(missing)}")

            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
            df.index = df.index.tz_localize(None)
            status.update(label="✅ 数据同步成功", state="complete")
            return df.rename(columns=targets).ffill().dropna(how='all')
        except Exception as e:
            status.update(label=f"❌ 引擎异常: {str(e)}", state="error")
            return pd.DataFrame()

# ================= 4. 回测逻辑 =================
@st.cache_data
def run_bt(df_all, assets, rs, rl, rw, h, m, user_start):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None, 0
    
    df_t = df_all[trade_names]
    rw_val = rw / 100.0
    scores = (df_t.pct_change(rs) * rw_val) + (df_t.pct_change(rl) * (1 - rw_val))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav, hist, t_count = np.ones(len(df_t)), [[] for _ in range(len(df_t))], 0
    s_v, p_v, m_v, r_v = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        mask = (s_v[i] > 0) & (p_v[i] > m_v[i])
        day_pnl, curr_h = 0.0, []
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_v[i][idx])[-h:]]
            day_pnl = np.nanmean(r_v[i+1][top_idx])
            curr_h = sorted([trade_names[j] for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
        if i > 0 and hist[i+1] != hist[i]: t_count += 1
            
    full = pd.DataFrame({"nav": nav, "h": hist}, index=df_t.index)
    final = full[full.index >= pd.Timestamp(user_start)].copy()
    if not final.empty: final['nav'] = final['nav'] / final['nav'].iloc[0]
    return final, scores, ma, df_t, t_count

# ================= 5. 渲染 =================
st.title("🏭 全球动能工厂")

if isinstance(dr, tuple) and len(dr) == 2:
    s_d, e_d = dr
    df = fetch_data_with_debug(st.session_state.my_assets, s_d, e_d, max(st.session_state.rs, st.session_state.rl, st.session_state.m))

    if not df.empty:
        bt_res = run_bt(df, st.session_state.my_assets, st.session_state.rs, st.session_state.rl, st.session_state.rw, st.session_state.h, st.session_state.m, s_d)
        res_df, score_df, ma_df, df_trade, t_count = bt_res if bt_res[0] is not None else (None, None, None, None, 0)
        
        if res_df is not None and not res_df.empty:
            nav = res_df['nav']
            
            # --- KPI ---
            mdd = ((nav - nav.cummax()) / nav.cummax()).min()
            k1, k2, k3 = st.columns(3)
            k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
            k2.metric("最大回撤", f"{mdd:.2%}", delta_color="inverse")
            k3.metric("调仓次数", f"{t_count} 次")

            # --- 图表 ---
            st.divider()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略净值", line=dict(color='#00ff88', width=3)))
            for b_name in BENCHMARKS.values():
                if b_name in df.columns:
                    b_nav = df[b_name][df.index >= pd.Timestamp(s_d)]
                    fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name=b_name, line=dict(dash='dot'), opacity=0.4))
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

            # --- 信号表 ---
            st.divider()
            st.subheader("📋 实时信号")
            l_s, l_p, l_m = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
            ranks = [{"名称": n, "评分": l_s[n], "信号": "✅ 持有" if (l_s[n]>0 and l_p[n]>l_m[n]) else "❌ 空仓"} for n in l_s.index]
            st.dataframe(pd.DataFrame(ranks).sort_values("评分", ascending=False).style.format({"评分": "{:.2%}"})
                         .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']), width="stretch")
else:
    st.info("💡 请在侧边栏完整选择回测区间（需点击两个日期）。")
