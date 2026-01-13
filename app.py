import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动能工厂 & 有效性检验", page_icon="🏭", layout="wide")

# ================= 侧边栏：参数控制区 =================
st.sidebar.header("🎛️ 因子合成实验室")

# 1. 动能因子构造 (ROC Parameter)
st.sidebar.subheader("1. 动能因子构造 (ROC)")
ROC_SHORT = st.sidebar.slider("短期 ROC 周期 (天)", 5, 60, 20, help="捕捉短期爆发力")
ROC_LONG = st.sidebar.slider("长期 ROC 周期 (天)", 30, 250, 60, help="捕捉中期趋势")
ROC_WEIGHT = st.sidebar.slider("短期权重 (%)", 0, 100, 100, help="100%表示只看短期，0%表示只看长期，50%表示各占一半") / 100.0

# 2. 交易参数
st.sidebar.subheader("2. 交易执行")
HOLD_COUNT = st.sidebar.number_input("持仓数量 (Top N)", min_value=1, max_value=5, value=1)
MA_EXIT = st.sidebar.number_input("止损均线 (MA)", min_value=5, max_value=120, value=20, help="跌破该均线强制空仓")

# 3. 回测设置
BACKTEST_START = st.sidebar.date_input("回测开始日期", datetime.date(2020, 1, 1))

# 标的池
ASSETS = {
    "513100": "纳指ETF",       
    "513520": "日经ETF",       
    "513180": "恒生科技",      
    "510180": "上证180",       
    "159915": "创业板指",      
    "518880": "黄金ETF",       
    "512400": "有色ETF",       
    "159981": "能源ETF",       
    "588050": "科创50",        
    "501018": "南方原油",      
}
BENCHMARKS = {"510300": "沪深300"}

# ================= 核心计算逻辑 =================

@st.cache_data(ttl=43200) 
def get_historical_data(start_date):
    """获取数据"""
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")
    
    targets = {**ASSETS, **BENCHMARKS}
    progress = st.empty()
    
    for i, (code, name) in enumerate(targets.items()):
        progress.text(f"正在加载: {name}...")
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date, adjust="qfq")
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')[['close']]
            df.columns = [name]
            
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
        except: pass
    
    progress.empty()
    return combined_df.sort_index().fillna(method='ffill')

def calculate_factors(df, roc_s, roc_l, w_s):
    """计算复合因子"""
    trade_cols = list(ASSETS.values())
    df_trade = df[trade_cols]
    
    # 计算双频 ROC
    roc_short = df_trade.pct_change(roc_s)
    roc_long = df_trade.pct_change(roc_l)
    
    # 合成得分
    score = roc_short * w_s + roc_long * (1 - w_s)
    
    # 均线
    ma_exit = df_trade.rolling(MA_EXIT).mean()
    
    return score, ma_exit, df_trade

def run_backtest(df_trade, score_df, ma_df):
    """回测引擎"""
    start_idx = max(ROC_LONG, ROC_SHORT, MA_EXIT) + 1
    if start_idx >= len(df_trade): return None, None, None
    
    curve = [1.0]
    dates = [df_trade.index[start_idx]]
    pos_history = []
    
    ret_daily = df_trade.pct_change()
    
    # 用于有效性分析的数据
    factor_analysis_data = [] # 记录每天的: [排名, 次日收益]

    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        mas = ma_df.iloc[i]
        
        # --- 1. 交易逻辑 ---
        # 选出动能 > 0 且 价格 > 均线 的
        valid = scores[(scores > 0) & (prices > mas)]
        
        targets = []
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
            
        # 计算次日收益
        day_pnl = 0.0
        if targets:
            w = 1.0 / HOLD_COUNT # 简单等权
            rets = ret_daily.iloc[i+1][targets]
            day_pnl = rets.sum() * w
            pos_history.append(",".join(targets))
        else:
            pos_history.append("现金")
            
        curve.append(curve[-1] * (1 + day_pnl))
        dates.append(df_trade.index[i+1])
        
        # --- 2. 收集数据用于因子有效性分析 ---
        # 我们想知道：今天排第1名的，明天到底涨了没？排倒数第1的，明天是不是跌了？
        # 获取所有资产当天的排名 (降序，分值越高名次越靠前)
        # 注意：这里我们不考虑均线过滤，纯粹看因子本身的预测能力
        daily_rank = scores.rank(ascending=False, method='first') 
        next_day_ret = ret_daily.iloc[i+1]
        
        for asset in scores.index:
            if not np.isnan(scores[asset]) and not np.isnan(next_day_ret[asset]):
                factor_analysis_data.append({
                    "Rank": int(daily_rank[asset]),
                    "Return": next_day_ret[asset]
                })

    return pd.Series(curve, index=dates), pos_history, pd.DataFrame(factor_analysis_data)

# ================= 主界面 =================

st.title("🏭 动能策略工厂")
st.markdown("通过调节 **ROC 参数**，观察策略变化，并检验因子是否有效。")

df_all = get_historical_data(BACKTEST_START)

if not df_all.empty:
    # 1. 计算
    score_df, ma_df, df_trade = calculate_factors(df_all, ROC_SHORT, ROC_LONG, ROC_WEIGHT)
    nav, history, factor_data = run_backtest(df_trade, score_df, ma_df)
    
    if nav is not None:
        # 创建两个标签页
        tab1, tab2 = st.tabs(["📈 策略回测", "🔬 因子有效性体检"])
        
        # ========== Tab 1: 回测结果 ==========
        with tab1:
            # 计算指标
            total_ret = (nav.iloc[-1] - 1) * 100
            cagr = (nav.iloc[-1] ** (365 / (nav.index[-1] - nav.index[0]).days) - 1) * 100
            drawdown = ((nav - nav.cummax()) / nav.cummax()).min() * 100
            
            # 纳指对比
            nasdaq = df_all['纳指ETF'].loc[nav.index[0]:]
            nasdaq = nasdaq / nasdaq.iloc[0]
            nasdaq_ret = (nasdaq.iloc[-1] - 1) * 100
            
            st.write("### 核心业绩")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益率", f"{total_ret:.1f}%", delta=f"{total_ret - nasdaq_ret:.1f}% vs 纳指")
            k2.metric("年化收益", f"{cagr:.1f}%")
            k3.metric("最大回撤", f"{drawdown:.1f}%", help="越小越好")
            k4.metric("参数
