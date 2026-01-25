import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026旗舰版", page_icon="🏭", layout="wide")

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
    rs = st.slider("短期评分周期 (天)", 5, 60, value=st.session_state.rs, key="rs", on_change=update_url)
    rl = st.slider("长期评分周期 (天)", 30, 250, value=st.session_state.rl, key="rl", on_change=update_url)
    rw = st.slider("权重分配 (短期%)", 0, 100, value=st.session_state.rw, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, value=st.session_state.h, key="h", on_change=update_url)
    m = st.number_input("风控均线 (MA)", 5, 120, value=st.session_state.m, key="m", on_change=update_url)
    
    st.divider()
    # 修改：日期范围选择
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    date_range = st.date_input(
        "选择回测区间",
        value=(default_start, today),
        max_value=today,
        key="date_range"
    )

# ================= 3. 数据与回测引擎 =================
@st.cache_data(ttl=3600)
def get_clean_data(assets_dict, start_date, end_date, warm_up_days):
    # 为了让回测第一天就有信号，实际抓取日期需要提前
    actual_fetch_start = start_date - datetime.timedelta(days=warm_up_days * 1.5 + 20)
    targets = {**assets_dict, **BENCHMARKS}
    try:
        data = yf.download(list(targets.keys()), start=actual_fetch_start, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        else:
            df = data
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except: return pd.DataFrame()

@st.cache_data
def run_enhanced_backtest(df_all, assets, rs, rl, rw, h, m, user_start_date):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None, 0
    
    df_t = df_all[trade_names]
    # 评分与均线计算
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    # 确定回测开始的索引位置（用户选择的日期或其后第一个交易日）
    # 确保 warm_up 足够
    warm_up = max(rs, rl, m)
    
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    trade_count = 0

    s_vals, p_vals, m_vals, r_vals = scores.values, df_t.values, ma.values, rets.values

    # 循环从预热结束开始，但我们最后只切出用户需要的部分
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
    
    # 截取用户要求的区间
    final_res = full_res[full_res.index >= pd.Timestamp(user_start_date)].copy()
    if not final_res.empty:
        final_res['nav'] = final_res['nav'] / final_res['nav'].iloc[0] # 重新归一化起始点为1
    
    return final_res, scores, ma, df_t, trade_count

# ================= 4. UI 渲染 =================
st.title("🏭 全球动能工厂")

# 处理日期选择逻辑
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    warm_up_needed = max(rs, rl, m)
    
    df = get_clean_data(st.session_state.my_assets, start_date, end_date, warm_up_needed)

    if not df.empty:
        bt = run_enhanced_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m, start_date)
        res_df, score_df, ma_df, df_trade, t_count = bt if bt[0] is not None else (None, None, None, None, 0)
        
        if res_df is not None and not res_df.empty:
            nav = res_df['nav']
            
            # --- 指标卡 ---
            mdd = ((nav - nav.cummax()) / nav.cummax()).min()
            daily_rets = nav.pct_change().dropna()
            sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252)) if not daily_rets.empty else 0
            
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
            k2.metric("年化收益", f"{(nav.iloc[-1]**(365/max((nav.index[-1]-nav.index[0]).days,1))-1):.2%}")
            k3.metric("最大回撤", f"{mdd:.2%}", delta_color="inverse")
            k4.metric("夏普比率", f"{sharpe:.2f}")
            k5.metric("调仓次数", f"{t_count} 次")

            # --- 图表渲染 ---
            st.divider()
            st.subheader(f"📈 策略净值走势 ({start_date} 至 {end_date})")
            
            fig = go.Figure()
            # 趋势背景
            ma_nav = nav.rolling(min(10, len(nav))).mean()
            status = (nav >= ma_nav).astype(int)
            change_idx = np.where(status.diff().fillna(0) != 0)[0]
            segs = np.concatenate(([0], change_idx, [len(nav)-1]))
            for i in range(len(segs)-1):
                cl = "rgba(0, 255, 136, 0.06)" if status.iloc[segs[i+1]] == 1 else "rgba(255, 68, 68, 0.06)"
                fig.add_vrect(x0=nav.index[segs[i]], x1=nav.index[segs[i+1]], fillcolor=cl, line_width=0, layer="below")

            # 主曲线
            fig.add_trace(go.Scatter(
                x=nav.index, y=nav, name="动能策略", 
                line=dict(color='#00ff88', width=3),
                text=[f"当前持仓: {', '.join(h) if h else '空仓'}" for h in res_df['holdings']],
                hoverinfo="x+y+text"
            ))

            # 调仓标记
            re_dates = [res_df.index[i] for i in range(1, len(res_df)) if res_df['holdings'].iloc[i] != res_df['holdings'].iloc[i-1]]
            fig.add_trace(go.Scatter(
                x=re_dates, y=nav.loc[re_dates], mode='markers', name="调仓日",
                marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1, color='#00ff88')),
                hoverinfo="skip"
            ))

            # 基准对比
            for b_name in BENCHMARKS.values():
                if b_name in df.columns:
                    # 基准同样需要截取日期并重新归一化
                    b_nav = df[b_name][df.index >= pd.Timestamp(start_date)]
                    if not b_nav.empty:
                        b_nav = b_nav / b_nav.iloc[0]
                        fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b_name, 
                                                 line=dict(dash='dot', width=1.2), opacity=0.6))

            fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=10, b=10),
                              hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, width="stretch")

            # --- 榜单 ---
            st.divider()
            st.subheader("📋 最新日度评分排行")
            l_scores, l_prices, l_mas = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
            ranks = []
            for name in l_scores.index:
                sig = "✅ 持有" if (l_scores[name] > 0 and l_prices[name] > l_mas[name]) else "❌ 空仓"
                ranks.append({"名称": name, "动能评分": l_scores[name], "最新价格": l_prices[name], "均线参考": l_mas[name], "操作建议": sig})
            
            df_rank = pd.DataFrame(ranks).sort_values("动能评分", ascending=False)
            st.dataframe(df_rank.style.format({"动能评分": "{:.2%}", "最新价格": "{:.3f}"})
                         .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['操作建议']),
                         width="stretch")
        else:
            st.error("无法在所选区间内生成回测数据，请尝试拉长日期范围或检查
