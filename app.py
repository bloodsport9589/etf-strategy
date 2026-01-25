import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026旗舰版", page_icon="🏭", layout="wide")

# 初始化参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20, "cash_y": 2.0}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        try:
            st.session_state[key] = float(st.query_params.get(key, val))
        except:
            st.session_state[key] = val

# 标的池
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "518880.SS": "黄金ETF",
    "510300.SS": "沪深300", "511130.SS": "30年国债ETF"
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
        st.markdown("**添加/删除标的**")
        c1, c2 = st.columns([2, 1])
        nc = c1.text_input("代码", key="input_code", placeholder="如: AAPL")
        nn = c2.text_input("名称", key="input_name", placeholder="如: 苹果")
        if st.button("➕ 添加", width="stretch"):
            if nc and nn:
                st.session_state.my_assets[nc] = nn
                st.toast(f"已尝试添加 {nn}", icon="✅")
                st.rerun()
        
        st.divider()
        for code, name in list(st.session_state.my_assets.items()):
            cols = st.columns([3, 1])
            cols[0].write(f"{name} ({code})")
            if cols[1].button("❌", key=f"del_{code}"):
                del st.session_state.my_assets[code]
                st.rerun()
    
    st.divider()
    rs = st.slider("短期周期 (rs)", 5, 60, value=int(st.session_state.rs), key="rs", on_change=update_url)
    rl = st.slider("长期周期 (rl)", 30, 250, value=int(st.session_state.rl), key="rl", on_change=update_url)
    rw = st.slider("权重 (短期%)", 0, 100, value=int(st.session_state.rw), key="rw", on_change=update_url) / 100.0
    h = st.number_input("最大持仓数", 1, 10, value=int(st.session_state.h), key="h", on_change=update_url)
    m = st.number_input("均线止损 (MA)", 5, 120, value=int(st.session_state.m), key="m", on_change=update_url)
    cash_y = st.slider("现金年化 (%)", 0.0, 5.0, value=float(st.session_state.cash_y), step=0.1, key="cash_y", on_change=update_url)
    start_d = st.date_input("回测起点", datetime.date(2021, 1, 1))

# ================= 3. 数据与回测引擎 (修复核心错误) =================
@st.cache_data(ttl=3600)
def get_clean_data(assets_dict, start_date):
    targets = {**assets_dict, **BENCHMARKS}
    try:
        data = yf.download(list(targets.keys()), start=start_date, progress=False)
        if data.empty: return pd.DataFrame()
        df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        df.index = df.index.tz_localize(None)
        # 强制转换为 float 类型，防止 object 类型导致 pct_change 崩溃
        df = df.astype(float)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

@st.cache_data
def run_enhanced_backtest(df_all, assets, rs, rl, rw, h, m, cash_annual_rate):
    # 筛选出数据中真实存在的品种
    available_names = [n for n in assets.values() if n in df_all.columns]
    if not available_names: return None, None, None, None, 0
    
    # 提取并确保数据全是浮点数
    df_t = df_all[available_names].copy()
    
    # --- 关键修复：清洗无效列 ---
    df_t = df_t.select_dtypes(include=[np.number]) # 只保留数值列
    df_t = df_t.loc[:, (df_t.notnull().sum() > max(rs, rl))] # 剔除数据量不足的列
    
    if df_t.empty: return None, None, None, None, 0

    # 计算动能评分
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    daily_cash_rate = (1 + cash_annual_rate/100)**(1/252) - 1
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    trade_count = 0
    current_names = df_t.columns.tolist()

    s_vals, p_vals, m_vals, r_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        # 信号：评分 > 0 且 价格 > 均线
        mask = (s_vals[i] > 0) & (p_vals[i] > m_vals[i])
        valid_idx = np.where(mask & ~np.isnan(s_vals[i]))[0]
        
        day_pnl = 0.0
        curr_h = []
        
        if len(valid_idx) > 0:
            top_idx = valid_idx[np.argsort(s_vals[i][valid_idx])[-h:]]
            k = len(top_idx)
            asset_pnl = np.nanmean(r_vals[i+1][top_idx])
            day_pnl = (asset_pnl * (k / h)) + (daily_cash_rate * ((h - k) / h))
            curr_h = sorted([current_names[j] for j in top_idx])
            if k < h: curr_h.append(f"现金({(h-k)/h:.0%})")
        else:
            day_pnl = daily_cash_rate
            curr_h = ["100% 现金"]
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
        if hist[i+1] != hist[i]: trade_count += 1
            
    res = pd.DataFrame({"nav": nav, "holdings": hist}, index=df_t.index).iloc[warm_up:]
    return res, scores, ma, df_t, trade_count

# ================= 4. UI 渲染 =================
st.title("🏭 全球动能工厂")

df = get_clean_data(st.session_state.my_assets, start_d)

if not df.empty:
    bt = run_enhanced_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m, cash_y)
    res_df, score_df, ma_df, df_trade, t_count = bt if bt and bt[0] is not None else (None, None, None, None, 0)
    
    if res_df is not None:
        nav = res_df['nav']
        mdd = ((nav - nav.cummax()) / nav.cummax()).min()
        daily_rets = nav.pct_change().dropna()
        sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252)) if not daily_rets.empty else 0
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("年化收益", f"{(nav.iloc[-1]**(365/max((nav.index[-1]-nav.index[0]).days,1))-1):.2%}")
        k3.metric("最大回撤", f"{mdd:.2%}", delta_color="inverse")
        k4.metric("夏普比率", f"{sharpe:.2f}")
        k5.metric("调仓次数", f"{t_count} 次")

        st.divider()
        fig = go.Figure()
        # 背景
        ma_line = nav.rolling(10).mean()
        status = (nav >= ma_line.fillna(method='bfill')).astype(int)
        change_idx = np.where(status.diff().fillna(0) != 0)[0]
        segs = np.concatenate(([0], change_idx, [len(nav)-1]))
        for i in range(len(segs)-1):
            cl = "rgba(0, 255, 136, 0.06)" if status.iloc[segs[i+1]] == 1 else "rgba(255, 68, 68, 0.06)"
            fig.add_vrect(x0=nav.index[segs[i]], x1=nav.index[segs[i+1]], fillcolor=cl, line_width=0, layer="below")

        fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略+现金", line=dict(color='#00ff88', width=3),
                                 text=[f"持仓: {', '.join(h)}" for h in res_df['holdings']], hoverinfo="x+y+text"))
        
        for b in BENCHMARKS.values():
            if b in df.columns:
                b_nav = df[b].loc[nav.index[0]:]
                fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name=b, line=dict(dash='dot'), opacity=0.5))

        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("⚠️ 选中的品种在该日期范围内有效数据不足，请尝试添加其它代码或调整日期。")
else:
    st.error("无法加载行情，请检查代码拼写或网络。")
