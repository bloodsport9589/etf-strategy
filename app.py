import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-2026版", page_icon="🏭", layout="wide")

# 初始化策略参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        # 修复 2026 query_params 获取逻辑
        try:
            url_val = st.query_params.get(key, val)
        except:
            url_val = val
        st.session_state[key] = int(url_val)

# 初始化品种池
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

def update_url():
    st.query_params.update({k: st.session_state[k] for k in DEFAULTS.keys()})

# ================= 2. 侧边栏：品种管理 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    
    with st.expander("📝 品种管理", expanded=True):
        st.markdown("**添加新标的**")
        new_code = st.text_input("代码 (如 AAPL 或 513100.SS)", key="input_code")
        new_name = st.text_input("名称 (如 苹果 或 纳指ETF)", key="input_name")
        
        if st.button("➕ 确认添加", width="stretch"):
            if new_code and new_name:
                st.session_state.my_assets[new_code] = new_name
                st.toast(f"已添加: {new_name}")
                st.rerun()
            else:
                st.error("请完整填写代码和名称")
        
        st.divider()
        st.markdown("**当前池内品种**")
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
    st.subheader("参数设置")
    rs = st.slider("短期ROC (天)", 5, 60, value=st.session_state.rs, key="rs", on_change=update_url)
    rl = st.slider("长期ROC (天)", 30, 250, value=st.session_state.rl, key="rl", on_change=update_url)
    rw = st.slider("短期权重 (%)", 0, 100, value=st.session_state.rw, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, value=st.session_state.h, key="h", on_change=update_url)
    m = st.number_input("止损均线 (MA)", 5, 120, value=st.session_state.m, key="m", on_change=update_url)
    start_d = st.date_input("回测开始", datetime.date(2022, 1, 1)) # 默认日期稍微延后，减少抓取负担

# ================= 3. 高速数据引擎 (增加诊断逻辑) =================
@st.cache_data(ttl=3600)
def get_data_diagnose(assets_dict, start_date):
    benchmarks = {"510300.SS": "沪深300", "^GSPC": "标普500"}
    targets = {**assets_dict, **benchmarks}
    ticker_list = list(targets.keys())
    
    with st.status("正在从全球服务器抓取行情...", expanded=False) as status:
        try:
            data = yf.download(ticker_list, start=start_date, progress=False, timeout=30)
            if data.empty:
                status.update(label="❌ 抓取失败：返回数据为空", state="error")
                return pd.DataFrame()
            
            # 2026 处理 MultiIndex 的稳健方法
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.levels[0]:
                    df = data['Adj Close']
                else:
                    df = data['Close']
            else:
                df = data
            
            df.index = df.index.tz_localize(None)
            df = df.rename(columns=targets).ffill().dropna(how='all', axis=0)
            
            status.update(label=f"✅ 成功抓取 {len(df.columns)} 个标的的历史数据", state="complete")
            return df
        except Exception as e:
            status.update(label=f"⚠️ 引擎报错: {str(e)}", state="error")
            return pd.DataFrame()

# ================= 4. 回测逻辑 =================
@st.cache_data
def run_backtest(df_all, assets, rs, rl, rw, h, m):
    # 只回测当前池子里的品种
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None
    
    df_t = df_all[trade_names]
    # 评分 = 短期变化*权重 + 长期变化*(1-权重)
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    if len(df_t) <= warm_up: return None, None, None, None

    nav = np.ones(len(df_t))
    holdings_history = [[] for _ in range(len(df_t))] 
    score_vals, price_vals, ma_vals, ret_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        # 选股条件：评分>0 且 价格高于均线
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

# ================= 5. 主页面 UI =================
st.title("🏭 全球动能工厂")

# 诊断式数据获取
df = get_data_diagnose(st.session_state.my_assets, start_d)

if df.empty:
    st.error("### 😭 无法加载主页面")
    st.markdown("""
    **可能的原因：**
    1. **标的代码无效**：请检查侧边栏是否有代码输入错误（如 A 股需加 `.SS`）。
    2. **API 限制**：Yahoo Finance 接口暂时繁忙，请尝试刷新页面。
    3. **网络问题**：服务器无法访问国际行情源。
    
    **您可以尝试：** 点击侧边栏底部的 **“🔄 重置为默认”**。
    """)
else:
    # 运行回测
    bt_res = run_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m)
    res_df, score_df, ma_df, df_trade = bt_res if bt_res[0] is not None else (None, None, None, None)
    
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
            if buys.empty: st.warning("🛡️ 当前建议：全额避险")
            else:
                st.success(f"🚀 建议持仓: {', '.join(buys['名称'].tolist())}")
        with c2:
            st.dataframe(rank_df.style.format({"动能评分": "{:.2%}", "价格": "{:.2f}"})
                         .map(lambda x: 'color: #00ff88' if "✅" in str(x) else 'color: #ff4444', subset=['信号']),
                         width="stretch")

        # --- 策略图表 ---
        st.divider()
        st.subheader("📈 策略表现与调仓详情")
        fig = go.Figure()
        
        # 背景染色
        ma_nav = nav.rolling(10).mean().fillna(method='bfill')
        status = (nav >= ma_nav).astype(int)
        change_idx = np.where(status.diff().fillna(0) != 0)[0]
        segs = np.concatenate(([0], change_idx, [len(nav)-1]))
        for i in range(len(segs)-1):
            cl = "rgba(0, 255, 136, 0.08)" if status.iloc[segs[i+1]] == 1 else "rgba(255, 68, 68, 0.08)"
            fig.add_vrect(x0=nav.index[segs[i]], x1=nav.index[segs[i+1]], fillcolor=cl, line_width=0, layer="below")

        # 净值曲线
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略净值", line=dict(color='#00ff88', width=3),
                                 text=[f"持仓: {h}" for h in res_df['holdings']], hoverinfo="x+y+text"))
        
        # 调仓菱形
        re_dates, re_text = [], []
        for i in range(1, len(res_df)):
            if res_df['holdings'].iloc[i] != res_df['holdings'].iloc[i-1]:
                re_dates.append(res_df.index[i])
                re_text.append(f"组成分更替")
        
        fig.add_trace(go.Scatter(x=re_dates, y=nav.loc[re_dates], mode='markers', name="调仓",
                                 marker=dict(symbol='diamond', size=8, color='white')))

        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        # KPI
        mdd = ((nav - nav.cummax()) / nav.cummax()).min()
        k1, k2, k3 = st.columns(3)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("最大回撤", f"{mdd:.2%}")
        k3.metric("测试周期", f"{(nav.index[-1]-nav.index[0]).days} 天")
    else:
        st.warning("⚠️ 计算结果为空，请在侧边栏检查日期设置或增加标的数量。")

st.caption("注：2026 旗舰版 - 适配最新 Streamlit 渲染引擎")
