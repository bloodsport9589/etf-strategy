import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go

# ================= 页面配置 =================
st.set_page_config(page_title="全球动量轮动策略 Pro", page_icon="📈", layout="wide")

st.title("📈 全球资产动能轮动策略 Pro")
st.markdown("### 趋势跟随 | 动能轮动 | 均线风控")

# ================= 策略配置 =================
# 策略参数
HOLD_COUNT = 2          # 持仓数量
MOMENTUM_FAST = 20      # 20日涨幅
MOMENTUM_SLOW = 60      # 60日涨幅
MA_FILTER = 60          # 均线防守
BACKTEST_START = "20200101" # 回测开始时间 (考虑到部分ETF上市较晚，设为2020年较为稳妥)

# 标的池
ASSETS = {
    "513100": "纳指ETF",       # 美股
    "513520": "日经ETF",       # 日本
    "513180": "恒生科技",      # 港股
    "510180": "上证180",       # A股价值
    "159915": "创业板指",      # A股成长
    "518880": "黄金ETF",       # 商品避险
    "512400": "有色ETF",       # 周期
    "159981": "能源ETF",       # 能源
    "588050": "科创50",        # 硬科技
    "501018": "南方原油",      # 原油
}

# ================= 数据获取与处理 =================
@st.cache_data(ttl=43200) # 缓存12小时，避免每次刷新都重新拉取长数据
def get_historical_data():
    """拉取所有标的的历史数据并合并"""
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 显示进度条
    progress_text = "正在拉取历史数据进行回测..."
    my_bar = st.progress(0, text=progress_text)
    
    total_assets = len(ASSETS)
    valid_data = {}
    
    for i, (code, name) in enumerate(ASSETS.items()):
        try:
            # 获取较长历史数据
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=BACKTEST_START, end_date=end_date, adjust="qfq")
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df[['close']]
            df.columns = [name] # 列名改为资产名称
            
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
                
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            
        my_bar.progress((i + 1) / total_assets, text=f"正在获取 {name} 数据...")
        
    my_bar.empty()
    combined_df = combined_df.sort_index().fillna(method='ffill') # 填充停牌数据
    return combined_df

def calculate_strategy(df_close):
    """计算策略每日净值"""
    # 1. 计算动能因子
    # 收益率
    ret_daily = df_close.pct_change()
    
    # 动能指标 (20日 + 60日)
    mom_20 = df_close.pct_change(MOMENTUM_FAST)
    mom_60 = df_close.pct_change(MOMENTUM_SLOW)
    score_df = mom_20 * 0.6 + mom_60 * 0.4
    
    # 均线
    ma_60 = df_close.rolling(window=MA_FILTER).mean()
    
    # 2. 模拟回测循环 (简化版向量化回测)
    # 初始化资金曲线
    strategy_curve = [1.0] 
    dates = [df_close.index[MA_FILTER]] # 从数据足够的那天开始
    
    # 从第 N 天开始遍历 (为了有足够数据计算均线)
    start_idx = MA_FILTER
    
    # 记录每天持仓
    position_history = [] 

    for i in range(start_idx, len(df_close) - 1):
        # 今天的状态，决定明天的持仓
        today_date = df_close.index[i]
        current_scores = score_df.iloc[i]
        current_prices = df_close.iloc[i]
        current_ma = ma_60.iloc[i]
        
        # 筛选逻辑
        # 1. 价格在均线上
        trend_ok = current_prices > current_ma
        # 2. 动能 > 0 (可选，这里严格一点)
        mom_ok = current_scores > 0
        
        # 结合筛选
        valid_assets = current_scores[trend_ok & mom_ok]
        
        # 排序取前 N
        if not valid_assets.empty:
            targets = valid_assets.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
        else:
            targets = [] # 空仓
        
        # 计算次日收益 (简单均分仓位)
        next_day_ret_series = ret_daily.iloc[i+1]
        
        if len(targets) > 0:
            # 持仓资产的平均收益
            # 假设资金均分给选中的资产，如果选中1个就100%，2个就各50%
            daily_pnl = next_day_ret_series[targets].mean()
            position_history.append(",".join(targets))
        else:
            # 空仓 (假设持有现金，收益为0，或者算一点货币基金收益)
            daily_pnl = 0.0 
            position_history.append("现金")
            
        new_nav = strategy_curve[-1] * (1 + daily_pnl)
        strategy_curve.append(new_nav)
        dates.append(df_close.index[i+1])
        
    # 生成结果 DataFrame
    result_df = pd.DataFrame({
        "Date": dates,
        "Strategy": strategy_curve
    }).set_index("Date")
    
    return result_df, position_history, df_close

# ================= 主界面逻辑 =================

# 1. 获取数据
df_all = get_historical_data()

if not df_all.empty:
    
    # 2. 运行回测
    df_nav, pos_history, df_close = calculate_strategy(df_all)
    
    # 计算最新一天的信号 (用于展示今日建议)
    latest_close = df_close.iloc[-1]
    latest_ma = df_close.rolling(window=MA_FILTER).mean().iloc[-1]
    
    roc_20 = df_close.pct_change(MOMENTUM_FAST).iloc[-1]
    roc_60 = df_close.pct_change(MOMENTUM_SLOW).iloc[-1]
    latest_score = roc_20 * 0.6 + roc_60 * 0.4
    
    # 构建今日排名表
    rank_data = []
    for name in ASSETS.values():
        if name in latest_close:
            s = latest_score[name]
            p = latest_close[name]
            m = latest_ma[name]
            rank_data.append({
                "名称": name,
                "综合动能": s,
                "现价": p,
                "60日趋势": "✅ 上涨" if p > m else "❌ 下跌",
                "20日涨幅": roc_20[name]
            })
    
    rank_df = pd.DataFrame(rank_data).sort_values(by="综合动能", ascending=False).reset_index(drop=True)
    
    # ================= 布局显示 =================
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("💡 今日交易信号")
        st.caption(f"数据日期: {df_all.index[-1].strftime('%Y-%m-%d')}")
        
        # 选出建议品种
        candidates = rank_df.head(HOLD_COUNT)
        buy_list = []
        for _, row in candidates.iterrows():
            if row['综合动能'] > 0 and row['60日趋势'] == "✅ 上涨":
                buy_list.append(row['名称'])
        
        if not buy_list:
            st.error("🛑 **空仓信号**\n\n市场全线走弱，建议持有现金或货币基金。")
        else:
            st.success(f"✅ **建议持仓**\n\n**{' + '.join(buy_list)}**")
            if len(buy_list) < HOLD_COUNT:
                st.info("⚠️ 部分仓位建议保持现金")

        st.markdown("---")
        st.markdown("**📊 实时动能排名 (Top 5)**")
        # 美化表格
        display_df = rank_df.head(5).copy()
        display_df['综合动能'] = display_df['综合动能'].apply(lambda x: f"{x*100:.2f}%")
        display_df['20日涨幅'] = display_df['20日涨幅'].apply(lambda x: f"{x*100:.2f}%")
        st.table(display_df[['名称', '综合动能', '60日趋势', '20日涨幅']])

    with col2:
        st.subheader("📈 策略历史回测 (2020至今)")
        
        # 计算总收益
        total_ret = (df_nav['Strategy'].iloc[-1] - 1) * 100
        # 绘制交互式图表
        fig = px.line(df_nav, x=df_nav.index, y='Strategy', title=f"策略净值曲线 (总回报: {total_ret:.2f}%)")
        
        # 添加装饰
        fig.update_layout(
            xaxis_title="",
            yaxis_title="净值 (起始=1)",
            hovermode="x unified",
            template="plotly_dark" # 使用深色主题
        )
        # 将最新的持仓显示在图表下方或其他地方
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("查看最近 10 次调仓记录"):
            # 显示最近10天的持仓历史
            history_df = pd.DataFrame({
                "日期": df_nav.index[-10:], 
                "持仓品种": pos_history[-10:]
            }).sort_values("日期", ascending=False)
            st.dataframe(history_df, use_container_width=True)

else:
    st.error("无法获取数据，请稍后再试或检查网络。")
