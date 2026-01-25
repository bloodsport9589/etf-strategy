import streamlit as st
import yfinance as yf  # 替换为 yfinance
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time

# ================= 页面配置 =================
st.set_page_config(page_title="全球动能工厂 (海外稳定版)", page_icon="🏭", layout="wide")

# ================= 初始默认标的池 =================
# 雅虎财经代码规则：沪市 .SS，深市 .SZ
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF",       
    "513520.SS": "日经ETF",       
    "513180.SS": "恒生科技",      
    "510180.SS": "上证180",       
    "159915.SZ": "创业板指",      
    "518880.SS": "黄金ETF",       
    "512400.SS": "有色ETF",       
    "159981.SZ": "能源ETF",       
    "588050.SS": "科创50",        
    "501018.SS": "南方原油",      
}
BENCHMARKS = {"510300.SS": "沪深300"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 侧边栏：控制台 =================
st.sidebar.header("🎛️ 策略控制台")

with st.sidebar.expander("📝 标的管理 (自定义)", expanded=False):
    st.write("注意：沪市代码加 .SS，深市加 .SZ")
    c1, c2 = st.columns([1, 1])
    new_code = c1.text_input("代码", placeholder="513330.SS")
    new_name = c2.text_input("名称", placeholder="恒生互联")
    
    if st.button("➕ 添加品种"):
        if "." in new_code and len(new_name) > 0:
            st.session_state.my_assets[new_code] = new_name
            st.rerun()
        else:
            st.error("请输入带后缀的代码 (如 .SS 或 .SZ)")

    st.divider()
    current_list = [f"{code} : {name}" for code, name in st.session_state.my_assets.items()]
    del_targets = st.multiselect("选择要删除的品种", current_list)
    
    if st.button("🗑️ 删除选中"):
        for item in del_targets:
            code = item.split(" : ")[0]
            if code in st.session_state.my_assets:
                del st.session_state.my_assets[code]
        st.rerun()

# --- 策略参数 ---
st.sidebar.subheader("1. 策略参数")
ROC_SHORT = st.sidebar.slider("短期 ROC (天)", 5, 60, 20)
ROC_LONG = st.sidebar.slider("长期 ROC (天)", 30, 250, 60)
ROC_WEIGHT = st.sidebar.slider("短期权重 (%)", 0, 100, 100) / 100.0
HOLD_COUNT = st.sidebar.number_input("持仓数量", min_value=1, max_value=10, value=1)
MA_EXIT = st.sidebar.number_input("止损均线 (MA)", min_value=5, max_value=120, value=20)
BACKTEST_START = st.sidebar.date_input("回测开始日期", datetime.date(2020, 1, 1))

# ================= 核心计算逻辑 =================

@st.cache_data(ttl=3600) 
def get_historical_data(start_date, asset_keys_tuple):
    """使用 yfinance 获取海外稳定的金融数据"""
    combined_df = pd.DataFrame()
    start_str = start_date.strftime("%Y-%m-%d")
    
    current_assets = st.session_state.my_assets
    targets = {**current_assets, **BENCHMARKS}
    
    progress = st.empty()
    total = len(targets)
    
    # 雅虎财经支持一次性下载多个代码，效率更高
    codes = list(targets.keys())
    try:
        progress.text(f"🚀 正在通过 Yahoo Finance 同步 {total} 个标的数据...")
        # 下载数据，使用线程提高速度
        data = yf.download(codes, start=start_str, interval="1d", progress=False)
        
        if 'Adj Close' in data:
            combined_df = data['Adj Close']
        elif 'Close' in data:
            combined_df = data['Close']
            
        # 将代码映射回中文名称
        combined_df = combined_df.rename(columns=targets)
        
    except Exception as e:
        st.error(f"数据获取失败: {e}")
    
    progress.empty()
    return combined_df.sort_index().ffill().bfill()

# --- 因子计算与回测逻辑 (保持不变) ---
def calculate_factors(df, roc_s, roc_l, w_s):
    trade_cols = list(st.session_state.my_assets.values())
    valid_cols = [c for c in trade_cols if c in df.columns]
    df_trade = df[valid_cols]
    roc_short = df_trade.pct_change(roc_s)
    roc_long = df_trade.pct_change(roc_l)
    score = roc_short * w_s + roc_long * (1 - w_s)
    ma_exit = df_trade.rolling(MA_EXIT).mean()
    return score, ma_exit, df_trade

def run_backtest(df_trade, score_df, ma_df):
    start_idx = max(ROC_LONG, ROC_SHORT, MA_EXIT) + 1
    if start_idx >= len(df_trade): return None, None
    curve = [1.0]; dates = [df_trade.index[start_idx]]
    ret_daily = df_trade.pct_change()
    factor_analysis_data = [] 

    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]; prices = df_trade.iloc[i]; mas = ma_df.iloc[i]
        valid = scores[(scores > 0) & (prices > mas)]
        targets = []
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
        day_pnl = 0.0
        if targets:
            w = 1.0 / HOLD_COUNT 
            valid_targets = [t for t in targets if t in ret_daily.columns]
            if valid_targets:
                rets = ret_daily.iloc[i+1][valid_targets]
                day_pnl = rets.sum() * w
        curve.append(curve[-1] * (1 + day_pnl))
        dates.append(df_trade.index[i+1])
        
        # 记录因子表现
        daily_rank = scores.rank(ascending=False, method='first') 
        next_day_ret = ret_daily.iloc[i+1]
        for asset in scores.index:
            r = daily_rank.get(asset); ret = next_day_ret.get(asset)
            if pd.notnull(r) and pd.notnull(ret):
                factor_analysis_data.append({"Rank": int(r), "Return": ret})
    return pd.Series(curve, index=dates), pd.DataFrame(factor_analysis_data)

# ================= 主界面展示 =================
st.title("🏭 全球动能工厂 (海外稳定版)")

asset_keys = tuple(sorted(st.session_state.my_assets.keys()))
df_all = get_historical_data(BACKTEST_START, asset_keys)

if not df_all.empty:
    score_df, ma_df, df_trade = calculate_factors(df_all, ROC_SHORT, ROC_LONG, ROC_WEIGHT)
    nav, factor_data = run_backtest(df_trade, score_df, ma_df)
    
    if nav is not None:
        # 今日信号模块
        st.divider()
        st.header("💡 今日实盘信号")
        latest_scores = score_df.iloc[-1]
        latest_prices = df_trade.iloc[-1]
        latest_mas = ma_df.iloc[-1]
        
        rank_data = []
        for name in latest_scores.index:
            s = latest_scores.get(name, -99); p = latest_prices.get(name, 0); m = latest_mas.get(name, 0)
            is_buy = (s > 0) and (p > m)
            rank_data.append({"名称": name, "综合动能": s, "现价": p, "止损线": m, "状态": "✅ 持有" if is_buy else "❌ 空仓"})
            
        df_rank = pd.DataFrame(rank_data).sort_values("综合动能", ascending=False).reset_index(drop=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("📢 建议操作")
            buys = df_rank[df_rank['状态'] == "✅ 持有"].head(HOLD_COUNT)
            if buys.empty: st.error("🛑 当前信号：全额空仓")
            else:
                st.success(f"✅ 建议买入 Top {HOLD_COUNT}")
                for n in buys['名称']: st.write(f"**{n}**")
        with c2:
            st.subheader("📊 实时动能排行")
            st.dataframe(df_rank.style.format({"综合动能": "{:.2%}", "现价": "{:.3f}", "止损线": "{:.3f}"}), use_container_width=True)

        # 表现分析模块
        st.divider()
        tab1, tab2 = st.tabs(["📈 策略净值回测", "🔬 因子有效性分析"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name='策略净值', line=dict(color='#00ff88', width=3)))
            # 基准对比
            if "纳指ETF" in df_all.columns:
                b_nav = df_all["纳指ETF"].loc[nav.index[0]:] 
                fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name='纳指ETF(基准)', line=dict(dash='dot')))
            fig.update_layout(template="plotly_dark", title="累计净值对比 (已扣除止损)")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            if not factor_data.empty:
                rank_perf = factor_data.groupby("Rank")["Return"].mean() * 100
                fig_bar = px.bar(x=rank_perf.index, y=rank_perf.values, title="分层排名与次日平均收益", color=rank_perf.values)
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("所选时间段内数据不足，无法回测")
else:
    st.error("数据加载失败，请检查网络或标的代码")
