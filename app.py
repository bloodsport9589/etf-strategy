import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-旗舰可视化版", page_icon="🏭", layout="wide")

# 初始化参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = int(st.query_params.get(key, val))

def update_url():
    st.query_params.update({k: st.session_state[k] for k in DEFAULTS.keys()})

# ================= 2. 标的池 =================
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 3. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    with st.expander("📝 品种管理", expanded=False):
        c1, c2 = st.columns([2, 1])
        nc = c1.text_input("代码", key="nc")
        nn = c2.text_input("名称", key="nn")
        if st.button("➕ 添加"):
            if nc and nn: 
                st.session_state.my_assets[nc] = nn
                st.rerun()
        
        st.divider()
        assets_items = list(st.session_state.my_assets.items())
        for code, name in assets_items:
            cols = st.columns([3, 1])
            cols[0].write(f"{name} ({code})")
            if cols[1].button("❌", key=f"del_{code}"):
                del st.session_state.my_assets[code]
                st.rerun()

        if st.button("🔄 重置品种"):
            st.session_state.my_assets = DEFAULT_ASSETS.copy()
            st.rerun()
            
    st.divider()
    rs = st.slider("短期ROC", 5, 60, value=st.session_state.rs, key="rs", on_change=update_url)
    rl = st.slider("长期ROC", 30, 250, value=st.session_state.rl, key="rl", on_change=update_url)
    rw = st.slider("短期权重(%)", 0, 100, value=st.session_state.rw, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, value=st.session_state.h, key="h", on_change=update_url)
    m = st.number_input("止损均线", 5, 120, value=st.session_state.m, key="m", on_change=update_url)
    start_d = st.date_input("回测开始", datetime.date(2020, 1, 1))

# ================= 4. 数据引擎 =================
@st.cache_data(ttl=3600)
def get_data(assets_dict, start_date):
    targets = {**assets_dict, **BENCHMARKS}
    try:
        data = yf.download(list(targets.keys()), start=start_date, progress=False, timeout=30)
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        else:
            df = data
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except: return pd.DataFrame()

# ================= 5. 回测引擎 =================
@st.cache_data
def run_backtest(df_all, assets, rs, rl, rw, h, m):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None
    
    df_t = df_all[trade_names]
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    holdings_history = [[] for _ in range(len(df_t))] 

    score_vals, price_vals, ma_vals, ret_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        mask = (s_row > 0) & (price_vals[i] > ma_vals[i])
        day_pnl = 0.0
        current_holdings = []
        
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_row[idx])[-h:]]
            day_pnl = np.nanmean(ret_vals[i+1][top_idx])
            current_holdings = [trade_names[j] for j in top_idx]
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        holdings_history[i+1] = current_holdings
    
    res = pd.DataFrame({"nav": nav, "holdings": holdings_history}, index=df_t.index).iloc[warm_up:]
    return res, scores, ma, df_t

# ================= 6. UI 渲染与增强图表 =================
st.title("🏭 全球动能工厂 (智能分析版)")
st.info("✅ 已修复语法错误。图表现在包含趋势背景色和调仓菱形标记。")

df = get_data(st.session_state.my_assets, start_d)

if not df.empty:
    res_df, score_df, ma_df, df_trade = run_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m)
    
    if res_df is not None:
        nav = res_df['nav']
        
        st.divider()
        st.subheader("📈 策略表现与调仓详情")
        
        fig = go.Figure()

        # 1. 趋势背景染色
        # 绿色: 净值 > 10日均线 (上升趋势); 红色: 净值 < 10日均线 (回调趋势)
        ma_nav = nav.rolling(10).mean().fillna(method='bfill')
        status = (nav >= ma_nav).astype(int) 
        change_idx = np.where(status.diff().fillna(0) != 0)[0]
        segments = np.concatenate(([0], change_idx, [len(nav)-1]))

        for i in range(len(segments)-1):
            start, end = segments[i], segments[i+1]
            is_up = status.iloc[end] == 1
            color = "rgba(0, 255, 136, 0.08)" if is_up else "rgba(255, 68, 68, 0.08)"
            fig.add_vrect(x0=nav.index[start], x1=nav.index[end], fillcolor=color, line_width=0, layer="below")

        # 2. 调仓标记逻辑
        rebalance_dates, rebalance_text = [], []
        for i in range(1, len(res_df)):
            prev, curr = set(res_df['holdings'].iloc[i-1]), set(res_df['holdings'].iloc[i])
            if prev != curr:
                rebalance_dates.append(res_df.index[i])
                added, removed = curr - prev, prev - curr
                text = f"<b>调仓详情:</b><br>+ {'/'.join(added) if added else '无'}<br>- {'/'.join(removed) if removed else '无'}"
                rebalance_text.append(text)

        # 3. 绘制主线
        hover_labels = [f"日期: {d.date()}<br>净值: {v:.4f}<br>持仓: {', '.join(h) if h else '空仓'}" 
                        for d, v, h in zip(res_df.index, nav, res_df['holdings'])]

        fig.add_trace(go.Scatter(
            x=nav.index, y=nav, name="策略净值",
            line=dict(color='#00ff88', width=3),
            text=hover_labels, hoverinfo="text"
        ))

        # 4. 绘制调仓标记点
        fig.add_trace(go.Scatter(
            x=rebalance_dates, y=nav.loc[rebalance_dates],
            mode='markers', marker=dict(symbol='diamond', size=9, color='white', line=dict(width=1, color='#00ff88')),
            name="调仓标记", text=rebalance_text, hoverinfo="text"
        ))

        # 5. 绘制基准
        for b in BENCHMARKS.values():
            if b in df.columns:
                b_nav = df[b].loc[nav.index[0]:]
                b_nav /= b_nav.iloc[0]
                fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b, line=dict(dash='dot', color='gray'), hoverinfo="skip"))

        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=10, b=10),
                          hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, width="stretch")

        # --- KPI 面板 ---
        # 修复此处括号问题
        mdd = ((nav - nav.cummax()) / nav.cummax()).min()
        cagr = (nav.iloc[-1]**(365/max((nav.index[-1]-nav.index[0]).days, 1)) - 1)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("年化收益率", f"{cagr:.2%}")
        k3.metric("最大回撤", f"{mdd:.2%}")
        k4.metric("调仓次数", f"{len(rebalance_dates)} 次")

        # --- 实时成分表 ---
        st.divider()
        latest_holdings = res_df['holdings'].iloc[-1]
        if latest_holdings:
            st.success(f"🚀 当前建议持仓阵容：{' | '.join(latest_holdings)}")
        else:
            st.warning("🛡️ 当前建议：全额避险（空仓状态）")
    else:
        st.error("回测计算失败，请尝试增加日期范围。")
else:
    st.error("📡 数据引擎未响应，请检查品种代码或网络。")
