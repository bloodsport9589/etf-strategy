import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动能工厂 (专业分析版)", page_icon="🏭", layout="wide")

# ================= 初始默认标的池 =================
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 侧边栏：控制台 =================
st.sidebar.header("🎛️ 策略控制台")
# (此处省略标的管理部分，保持与前版本一致)
# ... [保持之前的标的管理模块代码] ...

ROC_SHORT = st.sidebar.slider("短期 ROC (天)", 5, 60, 20)
ROC_LONG = st.sidebar.slider("长期 ROC (天)", 30, 250, 60)
ROC_WEIGHT = st.sidebar.slider("短期权重 (%)", 0, 100, 100) / 100.0
HOLD_COUNT = st.sidebar.number_input("持仓数量", min_value=1, max_value=10, value=1)
MA_EXIT = st.sidebar.number_input("止损均线 (MA)", min_value=5, max_value=120, value=20)
BACKTEST_START = st.sidebar.date_input("回测开始日期", datetime.date(2020, 1, 1))

# ================= 核心获取逻辑 =================
@st.cache_data(ttl=3600)
def get_historical_data(start_date, asset_keys_tuple):
    start_str = start_date.strftime("%Y-%m-%d")
    current_assets = st.session_state.my_assets
    targets = {**current_assets, **BENCHMARKS}
    try:
        data = yf.download(list(targets.keys()), start=start_str, progress=False)
        df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        df.index = df.index.tz_localize(None)
        df = df.rename(columns=targets).sort_index().ffill().dropna(how='all')
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return pd.DataFrame()

# ================= 增强版回测引擎 =================
def run_enhanced_backtest(df_all, roc_s, roc_l, w_s):
    # 提取交易标的
    trade_cols = [n for n in st.session_state.my_assets.values() if n in df_all.columns]
    df_trade = df_all[trade_cols]
    
    # 因子计算
    score_df = (df_trade.pct_change(roc_s) * w_s) + (df_trade.pct_change(roc_l) * (1-w_s))
    ma_df = df_trade.rolling(MA_EXIT).mean()
    ret_daily = df_trade.pct_change()
    
    warm_up = max(roc_s, roc_l, MA_EXIT)
    if len(df_trade) <= warm_up + 5: return None
    
    # 初始化
    nav = [1.0]
    dates = [df_trade.index[warm_up]]
    holdings_log = ["初始空仓"] # 存储每日持仓详情用于Hover
    
    for i in range(warm_up, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        mas = ma_df.iloc[i]
        
        # 选股逻辑
        valid = scores[(scores > 0) & (prices > mas)]
        day_pnl = 0.0
        daily_h_detail = "空仓现金"
        
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
            w = 1.0 / len(targets)
            rets = ret_daily.iloc[i+1][targets]
            day_pnl = rets.sum() * w
            
            # 生成Hover详情：品种,收盘价,当日涨跌
            detail_list = []
            for t in targets:
                p = df_trade.iloc[i+1][t]
                r = ret_daily.iloc[i+1][t]
                detail_list.append(f"{t}: {p:.2f} ({r:+.2%})")
            daily_h_detail = "<br>".join(detail_list)
            
        nav.append(nav[-1] * (1 + day_pnl))
        dates.append(df_trade.index[i+1])
        holdings_log.append(daily_h_detail)
        
    res = pd.DataFrame({"nav": nav, "holdings": holdings_log}, index=dates)
    return res

# ================= 主界面 =================
st.title("🏭 全球动能工厂 (专业回测版)")
asset_keys = tuple(sorted(st.session_state.my_assets.keys()))
df_all = get_historical_data(BACKTEST_START, asset_keys)

if not df_all.empty:
    bt_res = run_enhanced_backtest(df_all, ROC_SHORT, ROC_LONG, ROC_WEIGHT)
    
    if bt_res is not None:
        nav_series = bt_res['nav']
        
        # --- 图表绘制：分段颜色 ---
        fig = go.Figure()
        
        # 增加基准
        for b_name in BENCHMARKS.values():
            if b_name in df_all.columns:
                b_data = df_all[b_name].loc[nav_series.index[0]:]
                fig.add_trace(go.Scatter(x=b_data.index, y=b_data/b_data.iloc[0], name=b_name, line=dict(dash='dot', width=1)))

        # 策略曲线：利用 line.color 数组实现变色 (上升绿色，下降红色)
        # 注意：这里简化为点对点变色逻辑
        colors = ['#00ff88' if (nav_series.iloc[i] >= nav_series.iloc[i-1]) else '#ff4444' for i in range(len(nav_series))]
        
        fig.add_trace(go.Scatter(
            x=nav_series.index, y=nav_series,
            mode='lines+markers',
            name='策略净值',
            line=dict(width=2, color='#00ff88'), # 基础色
            marker=dict(size=4, color=colors), # 点位颜色反映当日涨跌
            customdata=bt_res['holdings'],
            hovertemplate="<b>日期: %{x}</b><br>净值: %{y:.3f}<br>当日持仓:<br>%{customdata}<extra></extra>"
        ))

        fig.update_layout(template="plotly_dark", hovermode="x unified", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # --- KPI 区域 ---
        total_days = (nav_series.index[-1] - nav_series.index[0]).days
        total_ret = (nav_series.iloc[-1] - 1)
        cagr = (nav_series.iloc[-1] ** (365 / total_days) - 1) if total_days > 0 else 0
        mdd = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("累计收益率", f"{total_ret:.2%}")
        c2.metric("年化收益率 (CAGR)", f"{cagr:.2%}", help="基于复利计算的年化增长率")
        c3.metric("最大回撤", f"{mdd:.2%}")

        # --- 今日信号 (保持之前版本逻辑) ---
        st.divider()
        st.subheader("📢 实时交易信号")
        # ... [此处放置今日排行的代码] ...
    else:
        st.warning("数据点过少，请调整时间范围。")
