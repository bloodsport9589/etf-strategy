import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置与安全初始化 =================
st.set_page_config(page_title="全球动能工厂-2026旗舰版", page_icon="🏭", layout="wide")

# 策略参数池
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}

# 强力初始化：确保 session_state 永远不会丢失键值
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        # 尝试从 URL 获取，失败则使用默认值
        try:
            url_val = st.query_params.get(key)
            st.session_state[key] = int(url_val) if url_val is not None else val
        except:
            st.session_state[key] = val

def update_url():
    """安全地同步参数到 URL"""
    new_params = {}
    for k in DEFAULTS.keys():
        # 只有当键确实存在时才读取，防止 KeyError
        if k in st.session_state:
            new_params[k] = str(st.session_state[k])
    st.query_params.update(new_params)

# 标的池
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 2. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    with st.expander("📝 品种管理", expanded=False):
        c1, c2 = st.columns([2, 1])
        nc, nn = c1.text_input("代码", key="inc"), c2.text_input("名称", key="inn")
        if st.button("➕ 添加标的", width="stretch"):
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
    # 绑定 session_state 的 Slider
    rs = st.slider("短期周期 (rs)", 5, 60, key="rs", on_change=update_url)
    rl = st.slider("长期周期 (rl)", 30, 250, key="rl", on_change=update_url)
    rw = st.slider("短期权重 %", 0, 100, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量 ($H$)", 1, 10, key="h", on_change=update_url)
    m = st.number_input("风控均线 ($MA$)", 5, 120, key="m", on_change=update_url)
    
    st.divider()
    st.subheader("📅 回测时间范围")
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    
    # 快捷按钮逻辑修复：直接操作 session_state 并触发 rerun
    col_q1, col_q2, col_q3 = st.columns(3)
    if col_q1.button("1年"): 
        st.session_state.d_range = (today - datetime.timedelta(days=365), today)
        st.rerun()
    if col_q2.button("3年"): 
        st.session_state.d_range = (today - datetime.timedelta(days=365*3), today)
        st.rerun()
    if col_q3.button("5年"): 
        st.session_state.d_range = (today - datetime.timedelta(days=365*5), today)
        st.rerun()
    
    dr = st.date_input("手动选择区间", value=st.session_state.get('d_range', (default_start, today)), key="d_range")

# ================= 3. 高效回测引擎 =================
@st.cache_data(ttl=3600)
def fetch_data(assets_dict, start_date, end_date, warm_up):
    actual_start = start_date - datetime.timedelta(days=warm_up * 1.6 + 20)
    targets = {**assets_dict, **BENCHMARKS}
    try:
        data = yf.download(list(targets.keys()), start=actual_start, end=end_date, progress=False)
        df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except: return pd.DataFrame()

@st.cache_data
def backtest_engine(df_all, assets, rs, rl, rw, h, m, user_start):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None, 0
    
    df_t = df_all[trade_names]
    # 计算 ROC 和 MA
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav, hist, t_count = np.ones(len(df_t)), [[] for _ in range(len(df_t))], 0
    s_v, p_v, m_v, r_v = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        # 绝对动能逻辑：$Price > MA$ 且 $Score > 0$
        mask = (s_v[i] > 0) & (p_v[i] > m_v[i])
        day_pnl, curr_h = 0.0, []
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_v[i][idx])[-h:]]
            day_pnl = np.nanmean(r_v[i+1][top_idx])
            curr_h = sorted([trade_names[j] for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
        if hist[i+1] != hist[i]: t_count += 1
            
    full_df = pd.DataFrame({"nav": nav, "h": hist}, index=df_t.index)
    final = full_df[full_df.index >= pd.Timestamp(user_start)].copy()
    if not final.empty: final['nav'] = final['nav'] / final['nav'].iloc[0]
    return final, scores, ma, df_t, t_count

# ================= 4. UI 渲染 =================
st.title("🏭 全球动能工厂")
st.info("✅ V8 稳定性补丁已部署：修复了参数同步导致的 KeyError 崩溃。")

if isinstance(dr, tuple) and len(dr) == 2:
    s_date, e_date = dr
    df = fetch_data(st.session_state.my_assets, s_date, e_date, max(rs, rl, m))

    if not df.empty:
        res_df, score_df, ma_df, df_trade, t_count = backtest_engine(df, st.session_state.my_assets, rs, rl, rw, h, m, s_date)
        
        if res_df is not None and not res_df.empty:
            nav = res_df['nav']
            
            # --- 指标面板 ---
            daily_r = nav.pct_change().dropna()
            mdd = ((nav - nav.cummax()) / nav.cummax()).min()
            sharpe = (daily_r.mean() * 252 - 0.02) / (daily_r.std() * np.sqrt(252)) if not daily_r.empty else 0
            
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
            k2.metric("年化收益", f"{(nav.iloc[-1]**(365/max((nav.index[-1]-nav.index[0]).days,1))-1):.2%}")
            k3.metric("最大回撤", f"{mdd:.2%}", delta_color="inverse")
            k4.metric("夏普比率", f"{sharpe:.2f}")
            k5.metric("调仓次数", f"{t_count} 次")

            # --- 图表美化 ---
            st.divider()
            fig = go.Figure()
            # 趋势背景
            ma_nav = nav.rolling(min(10, len(nav))).mean().fillna(method='bfill')
            status = (nav >= ma_nav).astype(int)
            c_idx = np.where(status.diff().fillna(0) != 0)[0]
            segs = np.concatenate(([0], c_idx, [len(nav)-1]))
            for i in range(len(segs)-1):
                cl = "rgba(0, 255, 136, 0.05)" if status.iloc[segs[i+1]] == 1 else "rgba(255, 68, 68, 0.05)"
                fig.add_vrect(x0=nav.index[segs[i]], x1=nav.index[segs[i+1]], fillcolor=cl, line_width=0, layer="below")

            # 策略主线
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name="动能策略", line=dict(color='#00ff88', width=3),
                                     text=[f"持仓: {', '.join(h) if h else '空仓'}" for h in res_df['h']], hoverinfo="x+y+text"))
            
            # 调仓点
            re_dates = [res_df.index[i] for i in range(1, len(res_df)) if res_df['h'].iloc[i] != res_df['h'].iloc[i-1]]
            fig.add_trace(go.Scatter(x=re_dates, y=nav.loc[re_dates], mode='markers', name="调仓日", 
                                     marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1, color='#00ff88'))))

            # 基准
            for b_name in BENCHMARKS.values():
                if b_name in df.columns:
                    b_nav = df[b_name][df.index >= pd.Timestamp(s_date)]
                    if not b_nav.empty:
                        fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name=b_name, 
                                                 line=dict(dash='dot', width=1.2), opacity=0.4))

            fig.update_layout(template="plotly_dark", height=550, margin=dict(l=10, r=10, t=10, b=10), 
                              hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, width="stretch")

            # --- 实时信号 ---
            st.divider()
            st.subheader("📋 实时信号明细")
            l_s, l_p, l_m = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
            ranks = [{"名称": n, "评分": l_s[n], "价格": l_p[n], "止损位": l_m[n], "信号": "✅ 持有" if (l_s[n]>0 and l_p[n]>l_m[n]) else "❌ 空仓"} for n in l_s.index]
            st.dataframe(pd.DataFrame(ranks).sort_values("评分", ascending=False).style.format({"评分": "{:.2%}", "价格": "{:.2f}"})
                         .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']), width="stretch")
        else:
            st.warning("⚠️ 无法在选定区间生成回测，请尝试拉长日期范围。")
    else:
        st.error("📡 数据引擎未响应，请检查品种代码是否规范。")
else:
    st.info("💡 请在侧边栏完整选择回测的【开始日期】和【结束日期】。")
