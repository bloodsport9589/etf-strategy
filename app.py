import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动量回测实验室", page_icon="🧪", layout="wide")

# ================= 侧边栏：参数控制区 =================
st.sidebar.header("🧪 策略参数实验室")
st.sidebar.markdown("调整参数，实时寻找最佳策略")

# 1. 核心参数
HOLD_COUNT = st.sidebar.slider("持仓数量 (Top N)", min_value=1, max_value=4, value=2, help="分散持仓可以降低波动，集中持仓进攻性更强")
MOMENTUM_WINDOW = st.sidebar.slider("动能窗口 (N日涨幅)", min_value=5, max_value=60, value=20, help="越小越灵敏，但噪音越大；越大越稳，但反应越慢")

# 2. 风控参数
st.sidebar.subheader("🛡️ 风控设置")
MA_EXIT = st.sidebar.slider("止损均线 (MA)", min_value=5, max_value=120, value=20, help="价格跌破该均线强制空仓。MA20适合短线，MA60适合长线")
MIN_HOLD_DAYS = st.sidebar.slider("最小持有天数 (防抖)", min_value=1, max_value=10, value=3, help="买入后至少持有N天，防止反复来回打脸")

# 3. 回测范围
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

BENCHMARKS_EXTRA = {"510300": "沪深300"}

# ================= 核心计算逻辑 =================

def calculate_max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def calculate_cagr(series):
    if len(series) < 1: return 0
    days = (series.index[-1] - series.index[0]).days
    if days == 0: return 0
    return (series.iloc[-1] / series.iloc[0]) ** (365 / days) - 1

def calculate_sharpe(series):
    """简单夏普比率 (假设无风险利率为0)"""
    if len(series) < 2: return 0
    ret = series.pct_change().dropna()
    return ret.mean() / ret.std() * np.sqrt(252)

@st.cache_data(ttl=43200) 
def get_historical_data(start_date_str):
    """获取数据 (带缓存)"""
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_str = start_date_str.strftime("%Y%m%d")
    
    # 进度条
    progress_text = st.empty()
    all_targets = {**ASSETS, **BENCHMARKS_EXTRA}
    total = len(all_targets)
    
    for i, (code, name) in enumerate(all_targets.items()):
        progress_text.text(f"正在加载数据: {name}...")
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
    
    progress_text.empty()
    return combined_df.sort_index().fillna(method='ffill')

def run_dynamic_backtest(df_close, hold_n, mom_win, ma_win, min_hold):
    """动态回测引擎"""
    trade_assets = list(ASSETS.values())
    valid_cols = [c for c in trade_assets if c in df_close.columns]
    df_trade = df_close[valid_cols]
    
    # 1. 计算因子
    ret_daily = df_trade.pct_change()
    score_df = df_trade.pct_change(mom_win) # 动态动能窗口
    ma_line = df_trade.rolling(window=ma_win).mean() # 动态均线
    
    # 2. 回测循环
    # 预热期取最大窗口
    start_idx = max(mom_win, ma_win)
    if start_idx >= len(df_trade): return pd.Series(), []
    
    strategy_curve = [1.0]
    dates = [df_trade.index[start_idx]]
    pos_history = [] 
    
    # 锁定状态记录 (用于最小持有期)
    # 格式: {asset_name: days_held}
    holding_days = {} 
    last_holdings = []

    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        ma_vals = ma_line.iloc[i]
        
        # --- 策略逻辑 ---
        
        # 1. 找出所有符合买入条件的 (动能>0 且 >均线)
        candidates = scores[(scores > 0) & (prices > ma_vals)].sort_values(ascending=False)
        potential_buys = candidates.index.tolist()
        
        current_targets = []
        
        # 2. 核心：结合最小持有期决定持仓
        # 先看昨天持有的，如果还没拿够天数，强制继续持有 (不管排名是否下降)
        locked_assets = []
        for asset in last_holdings:
            days = holding_days.get(asset, 0)
            if days < min_hold:
                # 检查是否触发硬止损 (比如暴跌)，如果严重破位也可以强制卖，这里暂只用均线
                # 如果还在均线上，就强制拿住
                if prices[asset] > ma_vals[asset]:
                    locked_assets.append(asset)
        
        # 填满剩余仓位
        slots_left = hold_n - len(locked_assets)
        new_picks = []
        
        if slots_left > 0:
            for asset in potential_buys:
                if asset not in locked_assets:
                    new_picks.append(asset)
                    if len(new_picks) == slots_left:
                        break
        
        current_targets = locked_assets + new_picks
        
        # 3. 更新持有天数
        new_holding_days = {}
        for asset in current_targets:
            # 如果昨天就有，天数+1；如果是新买的，天数=1
            new_holding_days[asset] = holding_days.get(asset, 0) + 1
        
        holding_days = new_holding_days
        last_holdings = current_targets
        
        # 4. 计算收益 (等权重)
        daily_pnl = 0.0
        if len(current_targets) > 0:
            w = 1.0 / hold_n # 哪怕只选出1个，也只占 1/N 仓位 (剩余现金)
            # w = 1.0 / len(current_targets) # 或者：选出几个就满仓几个 (更激进) -> 这里用保守算法，没选满就留现金
            
            rets = ret_daily.iloc[i+1][current_targets]
            daily_pnl = rets.sum() * w
            pos_history.append(",".join(current_targets))
        else:
            pos_history.append("现金")
            
        new_nav = strategy_curve[-1] * (1 + daily_pnl)
        strategy_curve.append(new_nav)
        dates.append(df_trade.index[i+1])

    return pd.Series(strategy_curve, index=dates), pos_history

# ================= 主界面 =================

st.title("🧪 策略实验室")
st.caption("拖动左侧滑块，找到纳指的克星。")

# 获取数据
df_all = get_historical_data(BACKTEST_START)

if not df_all.empty:
    # 运行回测
    nav, history = run_dynamic_backtest(df_all, HOLD_COUNT, MOMENTUM_WINDOW, MA_EXIT, MIN_HOLD_DAYS)
    
    if not nav.empty:
        # 基准处理
        b_nasdaq = df_all.get("纳指ETF")
        b_hs300 = df_all.get("沪深300")
        
        start_dt = nav.index[0]
        # 截取同时间段并归一化
        def prep_bench(s):
            if s is None: return None
            s = s.loc[start_dt:]
            return s / s.iloc[0]
        
        b_nasdaq = prep_bench(b_nasdaq)
        b_hs300 = prep_bench(b_hs300)
        
        # 计算指标
        s_cagr = calculate_cagr(nav)
        s_dd = calculate_max_drawdown(nav)
        s_sharpe = calculate_sharpe(nav)
        
        n_cagr = calculate_cagr(b_nasdaq) if b_nasdaq is not None else 0
        n_dd = calculate_max_drawdown(b_nasdaq) if b_nasdaq is not None else 0
        n_sharpe = calculate_sharpe(b_nasdaq) if b_nasdaq is not None else 0
        
        # --- KPI 展示 ---
        st.subheader("📊 回测结果对比")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("年化收益 (CAGR)", f"{s_cagr*100:.1f}%", delta=f"{(s_cagr-n_cagr)*100:.1f}% vs 纳指")
        col2.metric("最大回撤", f"{s_dd*100:.1f}%", delta=f"{-(n_dd-s_dd)*100:.1f}% vs 纳指", delta_color="inverse")
        col3.metric("夏普比率 (性价比)", f"{s_sharpe:.2f}", delta=f"{s_sharpe-n_sharpe:.2f}", help="越高越好，表示承受单位风险获得的超额回报")
        col4.metric("持仓数量", f"{HOLD_COUNT} 只")
        
        # --- 图表 ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav.index, y=nav, mode='lines', name='当前策略', line=dict(color='#00ff88', width=2)))
        if b_nasdaq is not None:
            fig.add_trace(go.Scatter(x=b_nasdaq.index, y=b_nasdaq, mode='lines', name='纳指100', line=dict(color='#3366ff', width=1.5)))
        if b_hs300 is not None:
            fig.add_trace(go.Scatter(x=b_hs300.index, y=b_hs300, mode='lines', name='沪深300', line=dict(color='#ff3333', width=1.5, dash='dot')))
        
        fig.update_layout(template="plotly_dark", hovermode="x unified", title="净值曲线", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 信号展示 ---
        st.divider()
        st.subheader("💡 基于当前参数的最新建议")
        
        # 重算今日信号
        trade_df = df_all[list(ASSETS.values())]
        scores = trade_df.pct_change(MOMENTUM_WINDOW).iloc[-1]
        prices = trade_df.iloc[-1]
        mas = trade_df.rolling(MA_EXIT).mean().iloc[-1]
        
        df_rank = pd.DataFrame({
            "名称": ASSETS.values(),
            "动能": [scores.get(n, -99) for n in ASSETS.values()],
            "现价": [prices.get(n, 0) for n in ASSETS.values()],
            "均线": [mas.get(n, 0) for n in ASSETS.values()]
        })
        
        # 筛选
        df_rank['状态'] = np.where((df_rank['动能']>0) & (df_rank['现价']>df_rank['均线']), '✅', '❌')
        df_rank = df_rank.sort_values("动能", ascending=False).reset_index(drop=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            candidates = df_rank[df_rank['状态']=='✅'].head(HOLD_COUNT)
            if candidates.empty:
                st.warning("🛑 建议空仓")
            else:
                st.success("✅ 建议持有")
                for _, row in candidates.iterrows():
                    st.write(f"**{row['名称']}** (动能: {row['动能']*100:.1f}%)")
        
        with c2:
            st.dataframe(df_rank.style.applymap(lambda v: 'color: #00ff88' if v=='✅' else 'color: #ff4444', subset=['状态']), use_container_width=True)

    else:
        st.warning("数据不足，请调整回测开始时间。")

else:
    st.error("无法加载数据。")
