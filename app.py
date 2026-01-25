import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026旗舰版", page_icon="🏭", layout="wide")

# 初始化参数 (增加了现金收益率参数 cash_y)
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20, "cash_y": 2}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        # 转换逻辑：优先看URL，没有看默认
        try:
            st.session_state[key] = float(st.query_params.get(key, val))
        except:
            st.session_state[key] = val

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
    st.subheader("核心策略参数")
    rs = st.slider("短期评分周期 (天)", 5, 60, value=int(st.session_state.rs), key="rs", on_change=update_url)
    rl = st.slider("长期评分周期 (天)", 30, 250, value=int(st.session_state.rl), key="rl", on_change=update_url)
    rw = st.slider("权重分配 (短期%)", 0, 100, value=int(st.session_state.rw), key="rw", on_change=update_url) / 100.0
    h = st.number_input("最大持仓数量", 1, 10, value=int(st.session_state.h), key="h", on_change=update_url)
    m = st.number_input("风控均线 (MA)", 5, 120, value=int(st.session_state.m), key="m", on_change=update_url)
    
    st.divider()
    st.subheader("💰 现金管理")
    cash_y = st.slider("现金模拟年化收益 (%)", 0.0, 5.0, value=float(st.session_state.cash_y), step=0.1, key="cash_y", on_change=update_url)
    start_d = st.date_input("回测起点", datetime.date(2021, 1, 1))

# ================= 3. 数据与回测引擎 =================
@st.cache_data(ttl=3600)
def get_clean_data(assets_dict, start_date):
    targets = {**assets_dict, **BENCHMARKS}
    try:
        data = yf.download(list(targets.keys()), start=start_date, progress=False)
        df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except: return pd.DataFrame()

@st.cache_data
def run_enhanced_backtest(df_all, assets, rs, rl, rw, h, m, cash_annual_rate):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None, 0
    
    df_t = df_all[trade_names]
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    # 计算每日现金收益率
    daily_cash_rate = (1 + cash_annual_rate/100)**(1/252) - 1
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    trade_count = 0

    s_vals, p_vals, m_vals, r_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        # 绝对动能过滤：评分>0 且 价格>均线
        mask = (s_vals[i] > 0) & (p_vals[i] > m_vals[i])
        
        # 选出的达标品种
        valid_idx = np.where(mask)[0]
        
        day_pnl = 0.0
        curr_h = []
        
        if len(valid_idx) > 0:
            # 按评分选出前 h 个
            top_idx = valid_idx[np.argsort(s_vals[i][valid_idx])[-h:]]
            k = len(top_idx) # 实际入选数量
            
            # 计算盈亏：(入选标的平均收益 * 权重) + (现金收益 * 剩余权重)
            asset_pnl = np.nanmean(r_vals[i+1][top_idx])
            day_pnl = (asset_pnl * (k / h)) + (daily_cash_rate * ((h - k) / h))
            
            curr_h = sorted([trade_names[j] for j in top_idx])
            if k < h:
                curr_h.append(f"现金模拟({((h-k)/h):.0%})")
        else:
            # 全不达标，全仓现金
            day_pnl = daily_cash_rate
            curr_h = ["100% 现金模拟"]
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
        if hist[i+1] != hist[i]: trade_count += 1
            
    res = pd.DataFrame({"nav": nav, "holdings": hist}, index=df_t.index).iloc[warm_up:]
    return res, scores, ma, df_t, trade_count

# ================= 4. UI 渲染 =================
st.title("🏭 全球动能工厂")
st.info(f"💡 现金策略已激活：当品种不满足动能条件时，仓位将获得年化 {cash_y}% 的现金收益。")

df = get_clean_data(st.session_state.my_assets, start_d)

if not df.empty:
    bt = run_enhanced_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m, cash_y)
    res_df, score_df, ma_df, df_trade, t_count = bt if bt[0] is not None else (None, None, None, None, 0)
    
    if res_df is not None:
        nav = res_df['nav']
        
        # --- 指标卡 ---
        mdd = ((nav - nav.cummax()) / nav.cummax()).min()
        daily_rets = nav.pct_change().dropna()
        # 夏普计算 (无风险利率设为 2%)
        sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252)) if not daily_rets.empty else 0
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("年化收益", f"{(nav.iloc[-1]**(365/max((nav.index[-1]-nav.index[0]).days,1))-1):.2%}")
        k3.metric("最大回撤", f"{mdd:.2%}", delta_color="inverse")
        k4.metric("夏普比率", f"{sharpe:.2f}")
        k5.metric("调仓次数", f"{t_count} 次")

        # --- 增强型 K 线图 ---
        st.divider()
        st.subheader("📈 策略净值走势 (含现金模拟)")
        
        fig = go.Figure()

        # A. 趋势背景
        ma_line = nav.rolling(10).mean()
        status = (nav >= ma_line).astype(int)
        change_idx = np.where(status.diff().fillna(0) != 0)[0]
        segs = np.concatenate(([0], change_idx, [len(nav)-1]))
        for i in range(len(segs)-1):
            cl = "rgba(0, 255, 136, 0.06)" if status.iloc[segs[i+1]] == 1 else "rgba(255, 68, 68, 0.06)"
            fig.add_vrect(x0=nav.index[segs[i]], x1=nav.index[segs[i+1]], fillcolor=cl, line_width=0, layer="below")

        # B. 策略主曲线 (更新 hover text)
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav, name="动能策略+现金", 
            line=dict(color='#00ff88', width=3),
            text=[f"持仓明细: {', '.join(h)}" for h in res_df['holdings']],
            hoverinfo="x+y+text"
        ))

        # C. 调仓标记点
        re_dates = [res_df.index[i] for i in range(1, len(res_df)) if res_df['holdings'].iloc[i] != res_df['holdings'].iloc[i-1]]
        fig.add_trace(go.Scatter(
            x=re_dates, y=nav.loc[re_dates], mode='markers', name="调仓日",
            marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1, color='#00ff88')),
            hoverinfo="skip"
        ))

        # D. 基准曲线
        for b_name in BENCHMARKS.values():
            if b_name in df.columns:
                b_nav = df[b_name].loc[nav.index[0]:]
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

        # --- 实时榜单 ---
        st.divider()
        st.subheader("📋 实时监控台")
        l_scores, l_prices, l_mas = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
        ranks = []
        for name in l_scores.index:
            is_buy = (l_scores[name] > 0 and l_prices[name] > l_mas[name])
            sig = "✅ 持有" if is_buy else "❌ 空仓(转入现金)"
            ranks.append({"名称": name, "动能评分": l_scores[name], "当前价格": l_prices[name], "均线止损": l_mas[name], "信号": sig})
        
        df_rank = pd.DataFrame(ranks).sort_values("动能评分", ascending=False)
        st.dataframe(df_rank.style.format({"动能评分": "{:.2%}", "当前价格": "{:.3f}"})
                     .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']),
                     width="stretch")
    else:
        st.error("数据不足，无法回测，请检查开始日期。")
else:
    st.warning("📡 正在同步全球行情数据...")
