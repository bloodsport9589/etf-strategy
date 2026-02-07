import streamlit as st
import yfinance as yf
import akshare as ak  # 核心新增：用于获取国内准确数据
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-混合数据版", page_icon="🏭", layout="wide")

# 初始化参数
DEFAULTS = {
    "rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20,
    "rsi_period": 14, 
    "rsi_limit": 80,    
    "acc_limit": -0.05 
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 默认资产池 (包含南方原油 501018)
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "豆粕ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油", # 此标的现在将通过 AkShare 获取
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 2. 核心计算逻辑 =================

def calculate_rsi_series(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

@st.cache_data(ttl=3600)
def get_clean_data(assets_dict, start_date, end_date):
    """
    混合数据获取逻辑：
    1. 国内标的 (数字开头) -> AkShare (东方财富接口，支持后复权，支持LOF)
    2. 国际标的 (非数字开头) -> YFinance
    """
    targets = {**assets_dict, **BENCHMARKS}
    
    # 日期格式化适配 AkShare
    s_date_str = (start_date - timedelta(days=365)).strftime("%Y%m%d")
    e_date_str = (end_date + timedelta(days=1)).strftime("%Y%m%d")
    
    combined_df = pd.DataFrame()
    
    # 进度条 (因为 AkShare 是串行请求，需要反馈)
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(targets)
    
    for i, (ticker, name) in enumerate(targets.items()):
        status_text.text(f"正在获取数据 ({i+1}/{total}): {name}...")
        progress_bar.progress((i + 1) / total)
        
        series_data = None
        
        try:
            # --- 分支 A: 国内基金/股票 (AkShare) ---
            if ticker[0].isdigit():
                code = ticker.split('.')[0] # 去掉后缀
                try:
                    # 获取 ETF/LOF 历史数据 (后复权)
                    df_ak = ak.fund_etf_hist_em(
                        symbol=code, 
                        period="daily", 
                        start_date=s_date_str, 
                        end_date=e_date_str, 
                        adjust="hfq"
                    )
                    if not df_ak.empty:
                        df_ak['date'] = pd.to_datetime(df_ak['日期'])
                        df_ak.set_index('date', inplace=True)
                        series_data = df_ak['收盘']
                except:
                    # 备用：如果是普通指数或股票
                    try:
                        df_ak = ak.stock_zh_a_hist(symbol=code, start_date=s_date_str, end_date=e_date_str, adjust="hfq")
                        df_ak['date'] = pd.to_datetime(df_ak['日期'])
                        df_ak.set_index('date', inplace=True)
                        series_data = df_ak['收盘']
                    except:
                        pass

            # --- 分支 B: 国际指数 (YFinance) ---
            else:
                # 针对 ^GSPC 等
                df_yf = yf.download(ticker, start=start_date - timedelta(days=365), end=end_date + timedelta(days=1), progress=False)
                if not df_yf.empty:
                    # 处理 MultiIndex
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        try:
                            series_data = df_yf[('Adj Close', ticker)]
                        except:
                            series_data = df_yf.iloc[:, 0] # 盲取第一列
                    else:
                        series_data = df_yf['Adj Close'] if 'Adj Close' in df_yf.columns else df_yf['Close']
                    
                    # 关键：去除时区，否则无法与 AkShare 数据合并
                    if series_data.index.tz is not None:
                        series_data.index = series_data.index.tz_localize(None)

            # --- 数据合并 ---
            if series_data is not None and not series_data.empty:
                series_data.name = ticker # 恢复原始 key 名字
                combined_df = pd.merge(combined_df, series_data, left_index=True, right_index=True, how='outer')
                
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue

    progress_bar.empty()
    status_text.empty()
    
    if combined_df.empty: return pd.DataFrame()

    # 清洗：重命名 -> 排序 -> 填充 -> 去空
    rename_map = {k: v for k, v in targets.items() if k in combined_df.columns}
    combined_df = combined_df.rename(columns=rename_map)
    combined_df = combined_df.sort_index().ffill().dropna(how='all')
    
    return combined_df

# ================= 3. 策略回测引擎 =================
def run_strategy_engine(df_all, assets, params, user_start_date, 
                        use_rsi_filter=False, use_acc_filter=False):
    
    rs, rl, rw = params['rs'], params['rl'], params['rw']
    h, m = params['h'], params['m']
    rsi_p, rsi_limit = params['rsi_period'], params['rsi_limit']
    acc_limit = params['acc_limit']

    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None
    
    df_t = df_all[trade_names]
    
    mom_short = df_t.pct_change(rs)
    mom_long = df_t.pct_change(rl)
    scores = (mom_short * rw) + (mom_long * (1-rw))
    
    rsi_df = df_t.apply(lambda x: calculate_rsi_series(x, rsi_p))
    acc_df = mom_short - mom_long 
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m, rsi_p)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    
    s_vals, p_vals, m_vals = scores.values, df_t.values, ma.values
    r_vals, rsi_vals, acc_vals = rets.values, rsi_df.values, acc_df.values
    
    filter_stats = {"rsi_triggered": 0, "acc_triggered": 0}

    for i in range(warm_up, len(df_t) - 1):
        valid_data = np.isfinite(s_vals[i]) & np.isfinite(p_vals[i]) & np.isfinite(m_vals[i])
        base_signal = (s_vals[i] > 0) & (p_vals[i] > m_vals[i])
        
        pass_rsi = (rsi_vals[i] < rsi_limit) if use_rsi_filter else True
        pass_acc = (acc_vals[i] > acc_limit) if use_acc_filter else True
        
        if use_rsi_filter and np.any(base_signal & ~pass_rsi): filter_stats['rsi_triggered'] += 1
        if use_acc_filter and np.any(base_signal & ~pass_acc): filter_stats['acc_triggered'] += 1

        final_mask = valid_data & base_signal & pass_rsi & pass_acc
        
        day_pnl = 0.0
        curr_h = []
        
        if np.any(final_mask):
            idx = np.where(final_mask)[0]
            top_idx = idx[np.argsort(s_vals[i][idx])[-h:]]
            day_pnl = np.nanmean(r_vals[i+1][top_idx])
            if np.isnan(day_pnl): day_pnl = 0.0
            curr_h = sorted([trade_names[j] for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
            
    res = pd.DataFrame({"nav": nav}, index=df_t.index)
    res['holdings'] = hist
    
    mask_slice = res.index >= pd.to_datetime(user_start_date)
    res = res.loc[mask_slice]
    if res.empty: return None

    res['nav'] = res['nav'] / res['nav'].iloc[0]
    
    return {
        "res": res, 
        "stats": filter_stats,
        "raw_scores": scores.loc[mask_slice],
        "raw_prices": df_t.loc[mask_slice],
        "raw_rsi": rsi_df.loc[mask_slice],
        "raw_acc": acc_df.loc[mask_slice],
        "raw_ma": ma.loc[mask_slice]
    }

# ================= 4. UI 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 实验参数设置")
    with st.expander("1. 基础动量参数", expanded=False):
        rs = st.slider("短期周期 (Fast)", 5, 60, 20)
        rl = st.slider("长期周期 (Slow)", 30, 250, 60)
        rw = st.slider("短期权重", 0, 100, 100) / 100.0
        h = st.number_input("持仓数", 1, 10, 1)
        m = st.number_input("风控均线 (MA)", 5, 120, 20)

    st.markdown("### 2. 新因子调节 (A/B Test)")
    st.info("💡 勾选下方开关，策略将自动踢出不达标的品种，并顺延买入下一名。")
    use_rsi = st.checkbox("启用 RSI 熔断", value=False)
    rsi_limit = st.slider("RSI 上限", 50, 95, 80)
    
    use_acc = st.checkbox("启用 加速度 过滤", value=False)
    acc_limit = st.slider("加速度 下限", -0.2, 0.1, -0.05, 0.01)

    st.divider()
    col_d1, col_d2 = st.columns(2)
    start_d = col_d1.date_input("开始", datetime.date.today() - datetime.timedelta(days=365*3))
    end_d = col_d2.date_input("结束", datetime.date.today())

params = {
    "rs": rs, "rl": rl, "rw": rw, "h": h, "m": m,
    "rsi_period": 14, "rsi_limit": rsi_limit, "acc_limit": acc_limit
}

# ================= 5. 主界面 =================
st.title("🧪 动能工厂 - 持仓透视实验室 (混合数据版)")

df = get_clean_data(st.session_state.my_assets, start_d, end_d)

if not df.empty:
    with st.spinner("正在进行双轨回测..."):
        res_base = run_strategy_engine(df, st.session_state.my_assets, params, start_d, False, False)
        res_new = run_strategy_engine(df, st.session_state.my_assets, params, start_d, use_rsi, use_acc)

    if res_base and res_new:
        nav_base = res_base['res']['nav']
        nav_new = res_new['res']['nav']
        
        # --- 顶部数据 ---
        def calc_metrics(nav):
            ret = nav.iloc[-1] - 1
            mdd = ((nav - nav.cummax()) / nav.cummax()).min()
            dr = nav.pct_change().dropna()
            shp = (dr.mean()*252)/(dr.std()*np.sqrt(252)) if dr.std()!=0 else 0
            return ret, mdd, shp

        rb, mb, sb = calc_metrics(nav_base)
        rn, mn, sn = calc_metrics(nav_new)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("累计收益 (优化后)", f"{rn:.2%}", delta=f"{rn-rb:.2%}")
        c2.metric("最大回撤", f"{mn:.2%}", delta=f"{mn-mb:.2%}", delta_color="inverse")
        c3.metric("夏普比率", f"{sn:.2f}", delta=f"{sn-sb:.2f}")
        c4.metric("当前策略持仓", ", ".join(res_new['res']['holdings'].iloc[-1]) if res_new['res']['holdings'].iloc[-1] else "空仓")

        # --- 图表区 ---
        tab1, tab2 = st.tabs(["📈 净值曲线", "🧬 详细持仓诊断"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav_base.index, y=nav_base, name="原始策略", line=dict(color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=nav_new.index, y=nav_new, name="当前策略", line=dict(color='#00ff88', width=3)))
            fig.update_layout(height=500, template="plotly_dark", title="A/B 测试净值对比")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### 🔎 截止回测结束日的持仓快照")
            st.caption("下表展示了所有标的的状态，以及**为什么**它们被买入或被剔除。")
            
            # 获取最后一天的数据
            last_idx = -1
            r_score = res_new['raw_scores'].iloc[last_idx]
            r_price = res_new['raw_prices'].iloc[last_idx]
            r_ma = res_new['raw_ma'].iloc[last_idx]
            r_rsi = res_new['raw_rsi'].iloc[last_idx]
            r_acc = res_new['raw_acc'].iloc[last_idx]
            
            # 获取真实的持仓列表 (这是最关键的修正)
            real_holdings = res_new['res']['holdings'].iloc[last_idx]
            
            snapshot = []
            for name in r_score.index:
                if pd.isna(r_score[name]) or pd.isna(r_price[name]): continue
                
                # 1. 基础硬指标
                is_above_ma = r_price[name] > r_ma[name]
                is_pos_score = r_score[name] > 0
                
                # 2. 软过滤指标
                rsi_ok = r_rsi[name] < rsi_limit
                acc_ok = r_acc[name] > acc_limit
                
                # 3. 判定状态
                if name in real_holdings:
                    status = "✅ 实际持仓"
                    reason = "综合排名第一且满足所有条件"
                    color_code = 1 # 绿
                else:
                    # 如果没持仓，分析原因
                    if not is_pos_score:
                        status = "⚪ 落选"
                        reason = "动能评分为负"
                        color_code = 0
                    elif not is_above_ma:
                        status = "⚪ 落选"
                        reason = "价格跌破均线"
                        color_code = 0
                    elif use_rsi and not rsi_ok:
                        status = "⛔ 熔断剔除"
                        reason = f"RSI({r_rsi[name]:.1f}) 超标"
                        color_code = -1 # 红
                    elif use_acc and not acc_ok:
                        status = "⛔ 熔断剔除"
                        reason = f"加速度({r_acc[name]:.1%}) 衰竭"
                        color_code = -1
                    else:
                        # 分数是正的，也没熔断，但没买 -> 说明排名不够高
                        status = "⚠️ 备选"
                        reason = "符合条件，但分数不是最高"
                        color_code = 2 # 黄
                        
                        # 特殊情况提示：如果没开启过滤，但指标很差
                        if (not use_rsi and not rsi_ok) or (not use_acc and not acc_ok):
                            reason += " (注意：指标已报警但未开启过滤)"

                snapshot.append({
                    "标的": name,
                    "动能评分": r_score[name],
                    "加速度": r_acc[name],
                    "RSI": r_rsi[name],
                    "🏛️ 实际持仓": status,
                    "📋 判定原因": reason,
                    "_code": color_code
                })
            
            df_snap = pd.DataFrame(snapshot).sort_values("动能评分", ascending=False)
            
            def color_row(val):
                if "持仓" in val: return 'color: #00ff88; font-weight: bold; background-color: rgba(0,255,136,0.1)'
                if "熔断" in val: return 'color: #ff4444; font-weight: bold'
                if "备选" in val: return 'color: #ffcc00'
                return 'color: gray'

            st.dataframe(
                df_snap.style.format({"动能评分": "{:.2%}", "加速度": "{:.2%}", "RSI": "{:.1f}"})
                .map(color_row, subset=['🏛️ 实际持仓']),
                use_container_width=True,
                height=600,
                column_config={
                    "_code": None # 隐藏辅助列
                }
            )
else:
    st.error("无法获取数据，请检查网络或 AkShare 接口状态")
