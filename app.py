import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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

# 会话状态初始化
if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 侧边栏：控制台 =================
st.sidebar.header("🎛️ 策略控制台")

with st.sidebar.expander("📝 标的管理 (自定义)", expanded=False):
    st.info("沪市代码加 .SS，深市加 .SZ")
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
    """从 yfinance 获取并清洗数据"""
    start_str = start_date.strftime("%Y-%m-%d")
    current_assets = st.session_state.my_assets
    targets = {**current_assets, **BENCHMARKS}
    codes = list(targets.keys())
    
    status = st.empty()
    status.text("🚀 正在同步全球行情数据...")
    
    try:
        # 下载数据
        data = yf.download(codes, start=start_str, progress=False)
        
        if data.empty:
            return pd.DataFrame()

        # 处理 yfinance 返回的多级索引 (Multi-index Columns)
        if isinstance(data.columns, pd.MultiIndex):
            # 优先取复权价 'Adj Close'，其次取 'Close'
            if 'Adj Close' in data.columns.levels[0]:
                df = data['Adj Close']
            else:
                df = data['Close']
        else:
            df = data[['Adj Close']] if 'Adj Close' in data.columns else data[['Close']]

        # 1. 移除时区信息 (重要：防止与 start_date 比较报错)
        df.index = df.index.tz_localize(None)
        
        # 2. 将代码映射回中文名称
        df = df.rename(columns=targets)
        
        # 3. 清洗：删除全是空值的行，并填充
        df = df.dropna(how='all').sort_index().ffill().bfill()
        
        status.empty()
        return df
    except Exception as e:
        st.error(f"⚠️ 数据同步失败: {e}")
        return pd.DataFrame()

def calculate_factors(df, roc_s, roc_l, w_s):
    trade_cols = list(st.session_state.my_assets.values())
    valid_cols = [c for c in trade_cols if c in df.columns]
    df_trade = df[valid_cols]
    
    # 动能计算
    roc_short = df_trade.pct_change(roc_s)
    roc_long = df_trade.pct_change(roc_l)
    score = roc_short * w_s + roc_long * (1 - w_s)
    
    # 止损均线
    ma_exit = df_trade.rolling(MA_EXIT).mean()
    
    return score, ma_exit, df_trade

def run_backtest(df_trade, score_df, ma_df):
    # 计算回测起点
    warm_up = max(ROC_LONG, ROC_SHORT, MA_EXIT)
    if len(df_trade) <= warm_up + 5:
        return None, None
        
    start_idx = warm_up + 1
    curve = [1.0]
    dates = [df_trade.index[start_idx]]
    ret_daily = df_trade.pct_change()
    factor_analysis_data = [] 

    # 模拟调仓循环
    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        mas = ma_df.iloc[i]
        
        # 筛选条件：动能 > 0 且 价格 > 止损线
        valid = scores[(scores > 0) & (prices > mas)]
        
        day_pnl = 0.0
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
            w = 1.0 / len(targets) # 等权分布
            # 获取次日涨跌幅
            rets = ret_daily.iloc[i+1][targets]
            day_pnl = rets.sum() * w
            
        curve.append(curve[-1] * (1 + day_pnl))
        dates.append(df_trade.index[i+1])
        
        # 收集因子统计数据
        daily_rank = scores.rank(ascending=False)
        next_day_ret = ret_daily.iloc[i+1]
        for asset in scores.index:
            r = daily_rank.get(asset)
            ret = next_day_ret.get(asset)
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
        # --- Part 1: 实盘信号 ---
        st.divider()
        st.header("💡 今日实盘信号")
        
        # 取最新一天的非空数据
        latest_scores = score_df.iloc[-1]
        latest_prices = df_trade.iloc[-1]
        latest_mas = ma_df.iloc[-1]
        
        rank_data = []
        for name in latest_scores.index:
            s = latest_scores.get(name, -99)
            p = latest_prices.get(name, 0)
            m = latest_mas.get(name, 0)
            is_buy = (s > 0) and (p > m)
            rank_data.append({
                "名称": name, 
                "综合动能": s, 
                "价格": p, 
                "MA止损线": m, 
                "信号": "✅ 持有" if is_buy else "❌ 空仓"
            })
            
        df_rank = pd.DataFrame(rank_data).sort_values("综合动能", ascending=False).reset_index(drop=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📢 操作建议")
            buys = df_rank[df_rank['信号'] == "✅ 持有"].head(HOLD_COUNT)
            if buys.empty:
                st.error("🛑 当前所有品种均走弱，建议全额空仓。")
            else:
                st.success(f"建议持有以下前 {len(buys)} 个标的：")
                for n in buys['名称']:
                    st.write(f"- **{n}**")
        
        with col2:
            st.subheader("📊 实时动能排行榜")
            st.dataframe(
                df_rank.style.format({"综合动能": "{:.2%}", "价格": "{:.3f}", "MA止损线": "{:.3f}"})
                .applymap(lambda v: 'color: #00ff88' if '✅' in str(v) else 'color: #ff4444', subset=['信号']),
                use_container_width=True
            )

        # --- Part 2: 深度回测 ---
        st.divider()
        tab1, tab2 = st.tabs(["📈 策略净值走势", "🔬 因子收益体检"])
        
        with tab1:
            fig = go.Figure()
            # 策略线
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name='策略净值', line=dict(color='#00ff88', width=3)))
            # 基准对比 (取沪深300)
            if "沪深300" in df_all.columns:
                bench = df_all["沪深300"].loc[nav.index[0]:]
                bench_nav = bench / bench.iloc[0]
                fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name='沪深300(基准)', line=dict(dash='dot', color='gray')))
            
            fig.update_layout(template="plotly_dark", hovermode="x unified", title="策略历史表现 (复利)")
            st.plotly_chart(fig, use_container_width=True)
            
            # KPI 指标
            total_ret = (nav.iloc[-1] - 1) * 100
            ann_ret = (nav.iloc[-1] ** (252/len(nav)) - 1) * 100
            mdd = ((nav - nav.cummax()) / nav.cummax()).min() * 100
            
            k1, k2, k3 = st.columns(3)
            k1.metric("累计收益率", f"{total_ret:.1f}%")
            k2.metric("年化收益率", f"{ann_ret:.1f}%")
            k3.metric("历史最大回撤", f"{mdd:.1f}%")

        with tab2:
            if not factor_data.empty:
                rank_perf = factor_data.groupby("Rank")["Return"].mean() * 100
                fig_bar = px.bar(
                    x=rank_perf.index, y=rank_perf.values, 
                    title="不同排名位置的次日平均涨跌幅",
                    labels={'x':'动能排名', 'y':'平均收益 (%)'},
                    color=rank_perf.values, color_continuous_scale="RdYlGn"
                )
                fig_bar.update_layout(template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("数据量不足，请尝试在左侧将'回测开始日期'提前。")
else:
    st.error("无法获取行情数据，请检查代码后缀是否正确 (如 .SS 或 .SZ)。")
