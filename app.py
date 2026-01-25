import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动能工厂-专业分析版", page_icon="📈", layout="wide")

# ================= 初始标的池 =================
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# (此处保持之前的侧边栏标的管理代码不变...)

# ================= 策略逻辑控制 =================
st.sidebar.subheader("策略参数")
ROC_SHORT = st.sidebar.slider("短期 ROC", 5, 60, 20)
ROC_LONG = st.sidebar.slider("长期 ROC", 30, 250, 60)
ROC_WEIGHT = st.sidebar.slider("短期权重 (%)", 0, 100, 100) / 100.0
HOLD_COUNT = st.sidebar.number_input("持仓数量", 1, 10, 1)
MA_EXIT = st.sidebar.number_input("止损均线", 5, 120, 20)
BACKTEST_START = st.sidebar.date_input("回测开始日期", datetime.date(2020, 1, 1))

@st.cache_data(ttl=3600)
def get_data(start_date, keys_tuple):
    start_str = start_date.strftime("%Y-%m-%d")
    targets = {**st.session_state.my_assets, **BENCHMARKS}
    data = yf.download(list(targets.keys()), start=start_str, progress=False)
    df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
    df.index = df.index.tz_localize(None)
    return df.rename(columns=targets).sort_index().ffill().dropna(how='all')

# ================= 核心回测引擎 =================
def run_pro_backtest(df_all):
    trade_cols = [n for n in st.session_state.my_assets.values() if n in df_all.columns]
    df_trade = df_all[trade_cols]
    
    score_df = (df_trade.pct_change(ROC_SHORT)*ROC_WEIGHT) + (df_trade.pct_change(ROC_LONG)*(1-ROC_WEIGHT))
    ma_df = df_trade.rolling(MA_EXIT).mean()
    ret_daily = df_trade.pct_change()
    
    warm_up = max(ROC_SHORT, ROC_LONG, MA_EXIT)
    nav = [1.0]; dates = [df_trade.index[warm_up]]; holdings = ["空仓"]
    
    for i in range(warm_up, len(df_trade) - 1):
        s, p, m = score_df.iloc[i], df_trade.iloc[i], ma_df.iloc[i]
        valid = s[(s > 0) & (p > m)]
        
        pnl = 0.0; h_text = "空仓现金"
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
            pnl = ret_daily.iloc[i+1][targets].mean()
            h_text = "<br>".join([f"{t}: {df_trade.iloc[i+1][t]:.2f}({ret_daily.iloc[i+1][t]:+.2%})" for t in targets])
            
        nav.append(nav[-1]*(1+pnl)); dates.append(df_trade.index[i+1]); holdings.append(h_text)
    
    return pd.DataFrame({"nav": nav, "holdings": holdings}, index=dates)

# ================= 主界面 =================
st.title("🏭 全球动能工厂 - 专业量化工作站")
df_all = get_data(BACKTEST_START, tuple(sorted(st.session_state.my_assets.keys())))

if not df_all.empty:
    res = run_pro_backtest(df_all)
    nav = res['nav']
    
    # --- 指标计算 ---
    days = (nav.index[-1] - nav.index[0]).days
    total_ret = nav.iloc[-1] - 1
    cagr = (nav.iloc[-1]**(365/days)-1)
    mdd = ((nav - nav.cummax())/nav.cummax()).min()
    
    # 夏普比率计算 (假设无风险利率 2%)
    daily_rets = nav.pct_change().dropna()
    sharpe = (daily_rets.mean() * 252 - 0.02) / (daily_rets.std() * np.sqrt(252))

    # --- 绘图 ---
    fig = go.Figure()

    # 1. 绘制背景变色块 (Shapes)
    for i in range(1, len(nav)):
        color = "rgba(0, 255, 136, 0.1)" if nav.iloc[i] >= nav.iloc[i-1] else "rgba(255, 68, 68, 0.1)"
        fig.add_vrect(
            x0=nav.index[i-1], x1=nav.index[i],
            fillcolor=color, layer="below", line_width=0,
        )

    # 2. 绘制净值曲线
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav, name="策略净值",
        line=dict(color='#00ff88', width=2.5),
        customdata=res['holdings'],
        hovertemplate="<b>日期: %{x}</b><br>净值: %{y:.3f}<br>持仓详情:<br>%{customdata}<extra></extra>"
    ))

    # 3. 基准对比
    for b in BENCHMARKS.values():
        if b in df_all.columns:
            b_nav = df_all[b].loc[nav.index[0]:]
            fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name=b, line=dict(dash='dot', width=1, color='gray')))

    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified",
                      xaxis_title="交易日期", yaxis_title="累计净值")
    st.plotly_chart(fig, use_container_width=True)

    # --- KPI 面板 ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("累计收益率", f"{total_ret:.2%}")
    k2.metric("年化收益 (CAGR)", f"{cagr:.2%}")
    k3.metric("夏普比率", f"{sharpe:.2f}", help="每承担一单位风险获得的超额回报")
    k4.metric("最大回撤", f"{mdd:.2%}")

else:
    st.error("无法加载数据，请检查网络或代码后缀。")
