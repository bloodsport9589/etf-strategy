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

# 默认资产池
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "豆粕ETF", "588050.SS": "科创50",
    "USO": "原油", 
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
    全能容错版数据获取：
    1. 优先尝试 AkShare (准确，后复权)
    2. 如果 AkShare 失败/为空，自动降级为 YFinance
    3. 确保列名最终统一为中文名称
    4. 强制对齐A股日历，剔除假期冗余数据
    """
    targets = {**assets_dict, **BENCHMARKS}
    
    fetch_start = start_date - timedelta(days=365) 
    s_date_str = fetch_start.strftime("%Y%m%d")
    e_date_str = (end_date + timedelta(days=1)).strftime("%Y%m%d")
    
    combined_df = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(targets)
    
    for i, (ticker, name) in enumerate(targets.items()):
        status_text.text(f"正在获取 ({i+1}/{total}): {name}...")
        progress_bar.progress((i + 1) / total)
        
        series_data = None
        
        # 尝试 1: AkShare (国内源，优先)
        if ticker[0].isdigit(): 
            try:
                code = ticker.split('.')[0]
                df_ak = ak.fund_etf_hist_em(
                    symbol=code, period="daily", start_date=s_date_str, end_date=e_date_str, adjust="hfq"
                )
                if not df_ak.empty:
                    df_ak['date'] = pd.to_datetime(df_ak['日期'])
                    df_ak.set_index('date', inplace=True)
                    series_data = df_ak['收盘']
            except:
                pass 

        # 尝试 2: YFinance (国际源，备用/降级)
        if series_data is None or series_data.empty:
            try:
                df_yf = yf.download(ticker, start=fetch_start, end=end_date + timedelta(days=1), progress=False)
                
                if not df_yf.empty:
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        try:
                            series_data = df_yf[('Adj Close', ticker)]
                        except:
                            series_data = df_yf.iloc[:, 0] 
                    else:
                        series_data = df_yf['Adj Close'] if 'Adj Close' in df_yf.columns else df_yf['Close']
                    
                    if series_data.index.tz is not None:
                        series_data.index = series_data.index.tz_localize(None)
            except Exception as e:
                print(f"Yahoo fetch failed for {ticker}: {e}")

        # 数据合并
        if series_data is not None and not series_data.empty:
            series_data.name = name 
            combined_df = pd.merge(combined_df, series_data, left_index=True, right_index=True, how='outer')
    
    progress_bar.empty()
    status_text.empty()

    if combined_df.empty:
        return pd.DataFrame()

    # ==========================================
    # 主日历对齐过滤（宏观规避节假日/周末）
    # ==========================================
    hs300_name = BENCHMARKS.get("510300.SS", "沪深300")
    if hs300_name in combined_df.columns:
        # 只保留沪深300有真实交易记录的日子，彻底干掉周末和A股节假日
        valid_a_share_dates = combined_df[hs300_name].dropna().index
        combined_df = combined_df.loc[valid_a_share_dates]

    # 向前填充停牌日的缺失值并清理空行
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
    
    # ==========================================
    # 微观停牌检测器
    # 逻辑：价格无波动判定为停牌/无交易，屏蔽其买入资格
    # ==========================================
    is_tradeable = (df_t.diff() != 0).fillna(True) 
    
    warm_up = max(rs, rl, m, rsi_p)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    
    s_vals, p_vals, m_vals = scores.values, df_t.values, ma.values
    r_vals, rsi_vals, acc_vals = rets.values, rsi_df.values, acc_df.values
    t_vals = is_tradeable.values
    
    filter_stats = {"rsi_triggered": 0, "acc_triggered": 0}

    for i in range(warm_up, len(df_t) - 1):
        valid_data = np.isfinite(s_vals[i]) & np.isfinite(p_vals[i]) & np.isfinite(m_vals[i])
        
        # 加入 t_vals[i] 判断，停牌标的当天直接失去买入资格
        base_signal = (s_vals[i] > 0) & (p_vals[i] > m_vals[i]) & t_vals[i]
        
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
        "raw_ma": ma.loc[mask_slice],
        "raw_tradeable": is_tradeable.loc[mask_slice] # 输出停牌状态
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

# ================= 5. 主界面 (诊断增强版) =================
st.title("🧪 动能工厂 - 持仓透视实验室 (调试版)")

debug_mode = st.checkbox("🐞 开启调试模式 (如果没图表请勾选此项)", value=True)

if debug_mode:
    st.info(f"正在获取数据... 时间范围: {start_d} 到 {end_d}")

# 1. 获取数据
df = get_clean_data(st.session_state.my_assets, start_d, end_d)

if df.empty:
    st.error("❌ 错误：无法获取任何数据。可能是网络问题或 AkShare/YFinance 在云端被拦截。")
else:
    if debug_mode:
        st.success(f"✅ 成功获取数据! 数据形状: {df.shape}")
        st.write("📊 原始数据前 3 行预览:", df.head(3))
        st.write("📋 包含的列名:", df.columns.tolist())

    # 2. 运行策略
    with st.spinner("正在进行双轨回测..."):
        res_base = run_strategy_engine(df, st.session_state.my_assets, params, start_d, False, False)
        res_new = run_strategy_engine(df, st.session_state.my_assets, params, start_d, use_rsi, use_acc)

    if res_base is None or res_new is None:
        st.warning("⚠️ 警告：策略计算返回了空值。")
        if res_base is None: st.write("❌ 原始策略结果为 None")
        if res_new is None: st.write("❌ 新策略结果为 None")
    
    # 3. 如果策略成功，则绘图
    else:
        nav_base = res_base['res']['nav']
        nav_new = res_new['res']['nav']
        
        def calc_metrics(nav):
            if len(nav) < 2: return 0, 0, 0 
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
        
        last_holdings = res_new['res']['holdings'].iloc[-1] if not res_new['res'].empty else []
        c4.metric("当前策略持仓", ", ".join(last_holdings) if last_holdings else "空仓")

        tab1, tab2 = st.tabs(["📈 净值曲线", "🧬 详细持仓诊断"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav_base.index, y=nav_base, name="原始策略", line=dict(color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=nav_new.index, y=nav_new, name="当前策略", line=dict(color='#00ff88', width=3)))
            fig.update_layout(height=500, template="plotly_dark", title="A/B 测试净值对比")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### 🔎 截止回测结束日的持仓快照")
            
            if not res_new['raw_scores'].empty:
                last_idx = -1
                r_score = res_new['raw_scores'].iloc[last_idx]
                r_price = res_new['raw_prices'].iloc[last_idx]
                r_ma = res_new['raw_ma'].iloc[last_idx]
                r_rsi = res_new['raw_rsi'].iloc[last_idx]
                r_acc = res_new['raw_acc'].iloc[last_idx]
                r_trad = res_new['raw_tradeable'].iloc[last_idx]
                
                real_holdings = res_new['res']['holdings'].iloc[last_idx]
                
                snapshot = []
                for name in r_score.index:
                    if name not in r_price.index or pd.isna(r_score[name]): continue
                    
                    is_above_ma = r_price[name] > r_ma[name]
                    is_pos_score = r_score[name] > 0
                    rsi_ok = r_rsi[name] < rsi_limit
                    acc_ok = r_acc[name] > acc_limit
                    
                    # 判定状态（优先判断停牌）
                    if not r_trad[name]:
                        status = "🚫 停牌熔断"
                        reason = "监测到价格无波动，判定停牌或未交易"
                        color_code = -2
                    elif name in real_holdings:
                        status = "✅ 实际持仓"
                        reason = "综合排名第一且满足所有条件"
                        color_code = 1 
                    else:
                        if not is_pos_score:
                            status = "⚪ 落选"
                            reason = "动能评分为负"
                            color_code = 0
                        elif not is_above_ma:
                            status = "⚪ 落选"
                            reason = "价格跌破均线"
                            color_code = 0
                        elif use_rsi and not rsi_ok:
                            status = "⛔ 指标剔除"
                            reason = f"RSI({r_rsi[name]:.1f}) 超标"
                            color_code = -1 
                        elif use_acc and not acc_ok:
                            status = "⛔ 指标剔除"
                            reason = f"加速度({r_acc[name]:.1%}) 衰竭"
                            color_code = -1
                        else:
                            status = "⚠️ 备选"
                            reason = "符合条件，但分数不是最高"
                            color_code = 2 
                            
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
                
                if snapshot:
                    df_snap = pd.DataFrame(snapshot).sort_values("动能评分", ascending=False)
                    
                    def color_row(val):
                        if "持仓" in val: return 'color: #00ff88; font-weight: bold; background-color: rgba(0,255,136,0.1)'
                        if "指标剔除" in val: return 'color: #ff4444; font-weight: bold'
                        if "停牌" in val: return 'color: #ffaa00; font-weight: bold; background-color: rgba(255,170,0,0.1)'
                        if "备选" in val: return 'color: #ffcc00'
                        return 'color: gray'

                    st.dataframe(
                        df_snap.style.format({"动能评分": "{:.2%}", "加速度": "{:.2%}", "RSI": "{:.1f}"})
                        .map(color_row, subset=['🏛️ 实际持仓']),
                        use_container_width=True,
                        height=600,
                        column_config={"_code": None}
                    )
                else:
                    st.info("没有生成快照数据")
            else:
                st.warning("结果集为空，无法显示详细诊断")
