import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026旗舰版", page_icon="🏭", layout="wide")

# 初始化策略参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = int(st.query_params.get(key, val))

# 初始化品种池
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
    
    with st.expander("📝 品种管理", expanded=True):
        st.markdown("**添加新标的**")
        new_code = st.text_input("代码 (如 AAPL 或 513100.SS)", key="input_code")
        new_name = st.text_input("名称 (如 苹果 或 纳指ETF)", key="input_name")
        if st.button("➕ 确认添加", width="stretch"):
            if new_code and new_name:
                st.session_state.my_assets[new_code] = new_name
                st.rerun()
        
        st.divider()
        current_assets = list(st.session_state.my_assets.items())
        for code, name in current_assets:
            cols = st.columns([3, 1])
            cols[0].write(f"{name}\n`{code}`")
            if cols[1].button("❌", key=f"del_{code}"):
                del st.session_state.my_assets[code]
                st.rerun()
        if st.button("🔄 重置为默认", width="stretch"):
            st.session_state.my_assets = DEFAULT_ASSETS.copy()
            st.rerun()
            
    st.divider()
    rs = st.slider("短期ROC (天)", 5, 60, value=st.session_state.rs, key="rs", on_change=update_url)
    rl = st.slider("长期ROC (天)", 30, 250, value=st.session_state.rl, key="rl", on_change=update_url)
    rw = st.slider("短期权重 (%)", 0, 100, value=st.session_state.rw, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, value=st.session_state.h, key="h", on_change=update_url)
    m = st.number_input("止损均线 (MA)", 5, 120, value=st.session_state.m, key="m", on_change=update_url)
    start_d = st.date_input("回测开始", datetime.date(2022, 1, 1))

# ================= 3. 数据引擎 (修复基准提取) =================
@st.cache_data(ttl=3600)
def get_data_v5(assets_dict, start_date):
    targets = {**assets_dict, **BENCHMARKS}
    ticker_list = list(targets.keys())
    
    with st.status("正在抓取行情数据...", expanded=False) as status:
        try:
            data = yf.download(ticker_list, start=start_date, progress=False, timeout=30)
            if data.empty: return pd.DataFrame()
            
            # 处理 2026 版 yfinance 多层索引
            if isinstance(data.columns, pd.MultiIndex):
                # 优先取 Adj Close，没有则取 Close
                df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
            else:
                df = data
            
            df.index = df.index.tz_localize(None)
            # 确保基准代码和自定义代码都能被重命名
            df = df.rename(columns=targets).ffill().dropna(how='all', axis=0)
            status.update(label=f"✅ 数据就绪 ({len(df.columns)} 个标的)", state="complete")
            return df
        except Exception as e:
            st.error(f"引擎报错: {e}")
            return pd.DataFrame()

# ================= 4. 回测逻辑 =================
@st.cache_data
def run_backtest_v5(df_all, assets, rs, rl, rw, h, m):
    # 策略交易品种（不含基准）
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None
    
    df_t = df_all[trade_names]
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    if len(df_t) <= warm_up: return None, None, None, None

    nav = np.ones(len(df_t))
    holdings_history = [[] for _ in range(len(df_t))] 
    rebalance_count = 0

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
            current_holdings = sorted([trade_names[j] for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        holdings_history[i+1] = current_holdings
        
        # 统计交易次数 (持仓组合变化即计为一次交易)
        if holdings_history[i+1] != holdings_history[i]:
            rebalance_count += 1
            
    res = pd.DataFrame({"nav": nav, "holdings": holdings_history}, index=df_t.index).iloc[warm_up:]
    return res, scores, ma, df_t, rebalance_count

# ================= 5. UI 渲染 =================
st.title("🏭 全球动能工厂")

df = get_data_v5(st.session_state.my_assets, start_d)

if not df.empty:
    bt_res = run_backtest_v5(df, st.session_state.my_assets, rs, rl, rw, h, m)
    res_df, score_df, ma_df, df_trade, trade_count = bt_res if bt_res[0] is not None else (None, None, None, None, 0)
    
    if res_df is not None:
        nav = res_df['nav']
        
        # --- 今日信号排位 ---
        st.divider()
        st.subheader("📊 今日实时信号与排位")
        l_scores, l_prices, l_mas = score_df.iloc[-1], df_trade.iloc[-1], ma_df.iloc[-1]
        
        ranks = []
        for name in l_scores.index:
            s, p, mv = l_scores[name], l_prices[name], l_mas[name]
            sig = "✅ 持有" if (s > 0 and p > mv) else "❌ 空仓"
            ranks.append({"名称": name, "动能评分": s, "价格": p, "止损线": mv, "信号": sig})
        
        rank_df = pd.DataFrame(ranks).sort_values("动能评分", ascending=False)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            buys = rank_df[rank_df['信号'] == "✅ 持有"].head(h)
            if buys.empty: st.warning("🛡️ 建议：全额避险")
            else: st.success(f"🚀 建议持仓: {', '.join(buys['名称'].tolist())}")
        with c2:
            st.dataframe(rank_df.style.format({"动能评分": "{:.2%}", "价格": "{:.2f}"})
                         .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']),
                         width="stretch")

        # --- 策略图表 (包含基准) ---
        st.divider()
        st.subheader("📈 策略表现与基准对比")
        fig = go.Figure()
        
        # 1. 净值曲线
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name="动能策略", line=dict(color='#00ff88', width=3),
                                 text=[f"持仓: {h}" for h in res_df['holdings']], hoverinfo="x+y+text"))
        
        # 2. 基准曲线 (修复逻辑)
        for b_code, b_name in BENCHMARKS.items():
            if b_name in df.columns:
                b_nav = df[b_name].loc[nav.index[0]:]
                b_nav = b_nav / b_nav.iloc[0] # 归一化
                fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=b_name, 
                                         line=dict(dash='dot', width=1.5), opacity=0.7))

        # 3. 调仓菱形
        re_dates = [res_df.index[i] for i in range(1, len(res_df)) if res_df['holdings'].iloc[i] != res_df['holdings'].iloc[i-1]]
        fig.add_trace(go.Scatter(x=re_dates, y=nav.loc[re_dates], mode='markers', name="调仓点",
                                 marker=dict(symbol='diamond', size=7, color='white')))

        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        # --- KPI 面板 (新增夏普和交易次数) ---
        mdd = ((nav - nav.cummax()) / nav.cummax()).min()
        
        # 计算夏普比率 (年化)
        daily_rets = nav.pct_change().dropna()
        if not daily_rets.empty:
            std = daily_rets.std()
            sharpe = (daily_rets.mean() * 252 - 0.02) / (std * np.sqrt(252)) if std != 0 else 0
        else:
            sharpe = 0
            
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("年化收益", f"{(nav.iloc[-1]**(365/max((nav.index[-1]-nav.index[0]).days, 1)) - 1):.2%}")
        k3.metric("最大回撤", f"{mdd:.2%}")
        k4.metric("夏普比率", f"{sharpe:.2f}")
        k5.metric("交易总次数", f"{trade_count} 次")
        
    else:
        st.warning("⚠️ 数据量不足，无法生成回测。")
else:
    st.error("无法加载数据，请检查网络或重置品种。")
