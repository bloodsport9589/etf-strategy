import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026视觉旗舰", page_icon="🏭", layout="wide")

# 初始化参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = int(st.query_params.get(key, val))

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

def update_url():
    st.query_params.update({k: st.session_state[k] for k in DEFAULTS.keys()})

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
    rs = st.slider("短期周期 (rs)", 5, 60, value=st.session_state.rs, key="rs", on_change=update_url)
    rl = st.slider("长期周期 (rl)", 30, 250, value=st.session_state.rl, key="rl", on_change=update_url)
    rw = st.slider("短期权重 %", 0, 100, value=st.session_state.rw, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, value=st.session_state.h, key="h", on_change=update_url)
    m = st.number_input("风控均线", 5, 120, value=st.session_state.m, key="m", on_change=update_url)
    
    st.divider()
    st.subheader("📅 回测时间范围")
    # 增加快捷日期选择
    col_q1, col_q2, col_q3 = st.columns(3)
    today = datetime.date.today()
    if col_q1.button("1年"): st.session_state.d_range = (today - datetime.timedelta(days=365), today)
    if col_q2.button("3年"): st.session_state.d_range = (today - datetime.timedelta(days=365*3), today)
    if col_q3.button("5年"): st.session_state.d_range = (today - datetime.timedelta(days=365*5), today)
    
    dr = st.date_input("手动选择区间", value=st.session_state.get('d_range', (today - datetime.timedelta(days=365), today)), key="date_input")

# ================= 3. 高效引擎 =================
@st.cache_data(ttl=3600)
def fetch_data(assets_dict, start_date, end_date, warm_up):
    # 预热期逻辑：额外抓取数据确保第一天就有 ROC 评分
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
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
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
        if hist[i+1] != hist[i]: t_count += 1
            
    full_df = pd.DataFrame({"nav": nav, "h": hist}, index=df_t.index)
    # 截取区间并归一化
    final = full_df[full_df.index >= pd.Timestamp(user_start)].copy()
    if not final.empty: final['nav'] = final['nav'] / final['nav'].iloc[0]
    return final, scores, ma, df_t, t_count

# ================= 4. 界面渲染 =================
st.title("🏭 全球动能工厂")

if isinstance(dr, tuple) and len(dr) == 2:
    s_date, e_date = dr
    df = fetch_data(st.session_state.my_assets, s_date, e_date, max(rs, rl, m))

    if not df.empty:
        res_df, score_df, ma_df, df_trade, t_count = backtest_engine(df, st.session_state.my_assets, rs, rl, rw, h, m, s_date)
        
        if res_df is not None and not res_df.empty:
            nav = res_df['nav']
            
            # --- KPI 指标 ---
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
            st.subheader(f"📊 策略路径分析 ({s_date} ➟ {e_date})")
            fig = go.Figure()

            # 1. 区域背景 (基于趋势判定)
            status = (nav >= nav.rolling(10).mean().fillna(method='bfill')).astype(int)
            c_idx = np.where(status.diff().fillna(0) != 0)[0]
            segs = np.concatenate(([0], c_idx, [len(nav)-1]))
            for i in range(len(segs)-1):
                cl = "rgba(0, 255, 136, 0.05)" if status.iloc[segs[i+1]] == 1 else "rgba(255, 68, 68, 0.05)"
                fig.add_vrect(x0=nav.index[segs[i]], x1=nav.index[segs[i+1]], fillcolor=cl, line_width=0, layer="below")

            # 2. 策略曲线 & 调仓点
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name="动能策略", line=dict(color='#00ff88', width=3),
                                     text=[f"持仓: {', '.join(h) if h else '空仓'}" for h in res_df['h']], hoverinfo="x+y+text"))
            
            re_dates = [res_df.index[i] for i in range(1, len(res_df)) if res_df['h'].iloc[i] != res_df['h'].iloc[i-1]]
            fig.add_trace(go.Scatter(x=re_dates, y=nav.loc[re_dates], mode='markers', name="调仓", 
                                     marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1, color='#00ff88'))))

            # 3. 基准曲线 (归一化)
            for b_name in BENCHMARKS.values():
                if b_name in df.columns:
                    b_nav = df[b_name][df.index >= pd.Timestamp(s_date)]
                    if not b_nav.empty:
                        fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name=b_name, 
                                                 line=dict(dash='dot', width=1.2), opacity=0.5))

            fig.update_layout(template="plotly_dark", height=550, margin=dict(l=10, r=10, t=10, b=10), 
                              hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, width="stretch")

            # --- 实时信号明细 ---
            st.divider()
            st.subheader("📋 实时评分与操作建议")
            l_s, l_p, l_m = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
            ranks = [{"名称": n, "动能分": l_s[n], "最新价": l_p[n], "止损位": l_m[n], "操作": "✅ 持有" if (l_s[n]>0 and l_p[n]>l_m[n]) else "❌ 空仓"} for n in l_s.index]
            st.dataframe(pd.DataFrame(ranks).sort_values("动能分", ascending=False).style.format({"动能分": "{:.2%}", "最新价": "{:.2f}"})
                         .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['操作']), width="stretch")
        else:
            st.error("无法生成回测，请检查日期范围是否包含足够的交易日。")
    else:
        st.warning("📡 数据抓取中，请确认代码正确且网络畅通。")
else:
    st.info("💡 请选择完整的开始与结束日期。")
