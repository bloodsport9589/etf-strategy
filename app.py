import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-因子实验室", page_icon="🧪", layout="wide")

# 初始化参数 (包含新因子默认值)
DEFAULTS = {
    "rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20,
    "rsi_period": 14, 
    "rsi_limit": 80,   # RSI 默认阈值
    "acc_limit": -0.05 # 加速度默认阈值
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 默认标的池
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
    "159985.SZ": "豆粕ETF",
}
BENCHMARKS = {"510300.SS": "沪深300", "^GSPC": "标普500"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 2. 核心计算逻辑 =================

# 计算 RSI 指标
def calculate_rsi_series(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# 数据下载与清洗
@st.cache_data(ttl=3600)
def get_clean_data(assets_dict, start_date, end_date):
    targets = {**assets_dict, **BENCHMARKS}
    # 多取一年数据用于指标预热
    fetch_start = start_date - timedelta(days=365)
    fetch_end = end_date + timedelta(days=1)
    
    try:
        data = yf.download(list(targets.keys()), start=fetch_start, end=fetch_end, progress=False, group_by='ticker')
        if data.empty: return pd.DataFrame()

        clean_data = pd.DataFrame()
        
        # 逐个提取有效列
        for ticker in targets.keys():
            try:
                if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                    col_data = data[ticker]
                elif ticker in data.columns:
                    col_data = data[[ticker]]
                else:
                    continue

                # 优先取 Adj Close
                if 'Adj Close' in col_data.columns:
                    s = col_data['Adj Close']
                elif 'Close' in col_data.columns:
                    s = col_data['Close']
                else:
                    s = col_data.iloc[:, 0]
                
                # 去除空列
                if s.dropna().empty: continue 
                clean_data[ticker] = s

            except Exception:
                continue 

        if clean_data.empty: return pd.DataFrame()
        
        # 重命名并格式化
        rename_map = {k: v for k, v in targets.items() if k in clean_data.columns}
        clean_data = clean_data.rename(columns=rename_map)
        clean_data.index = clean_data.index.tz_localize(None)
        clean_data = clean_data.ffill().dropna(how='all')
        
        return clean_data

    except Exception:
        return pd.DataFrame()

# ================= 3. 策略回测引擎 (支持 A/B Test) =================
# 这个引擎被设计为通用型，可以接受 use_rsi_filter 等开关
def run_strategy_engine(df_all, assets, params, user_start_date, 
                        use_rsi_filter=False, use_acc_filter=False):
    
    # 解包参数
    rs, rl, rw = params['rs'], params['rl'], params['rw']
    h, m = params['h'], params['m']
    rsi_p, rsi_limit = params['rsi_period'], params['rsi_limit']
    acc_limit = params['acc_limit']

    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None
    
    df_t = df_all[trade_names]
    
    # --- 1. 计算所有因子 ---
    mom_short = df_t.pct_change(rs)
    mom_long = df_t.pct_change(rl)
    scores = (mom_short * rw) + (mom_long * (1-rw))
    
    # 辅助因子
    rsi_df = df_t.apply(lambda x: calculate_rsi_series(x, rsi_p))
    acc_df = mom_short - mom_long # 加速度：短期 - 长期

    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m, rsi_p)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    
    # 转换为numpy以加速循环
    s_vals = scores.values
    p_vals = df_t.values
    m_vals = ma.values
    r_vals = rets.values
    rsi_vals = rsi_df.values
    acc_vals = acc_df.values
    
    # 统计拦截次数
    filter_stats = {"rsi_triggered": 0, "acc_triggered": 0}

    # --- 2. 逐日交易循环 ---
    for i in range(warm_up, len(df_t) - 1):
        # 基础数据有效性
        valid_data = np.isfinite(s_vals[i]) & np.isfinite(p_vals[i]) & np.isfinite(m_vals[i])
        
        # A. 基础信号：动能>0 且 价格>均线
        base_signal = (s_vals[i] > 0) & (p_vals[i] > m_vals[i])
        
        # B. 因子过滤 (如果是False则该标的被剔除)
        # 逻辑：如果不启用过滤器(False)，则默认为通过(True)；否则检查数值
        pass_rsi = (rsi_vals[i] < rsi_limit) if use_rsi_filter else True
        pass_acc = (acc_vals[i] > acc_limit) if use_acc_filter else True
        
        # 统计：仅当有基础信号却被新因子拦截时计数
        if use_rsi_filter and np.any(base_signal & ~pass_rsi): filter_stats['rsi_triggered'] += 1
        if use_acc_filter and np.any(base_signal & ~pass_acc): filter_stats['acc_triggered'] += 1

        # C. 最终候选池 = 数据有效 & 基础信号 & RSI通过 & 加速度通过
        final_mask = valid_data & base_signal & pass_rsi & pass_acc
        
        day_pnl = 0.0
        curr_h = []
        
        if np.any(final_mask):
            idx = np.where(final_mask)[0]
            # 从幸存者中，选动能分数最高的 Top H
            top_idx = idx[np.argsort(s_vals[i][idx])[-h:]]
            
            day_pnl = np.nanmean(r_vals[i+1][top_idx])
            if np.isnan(day_pnl): day_pnl = 0.0
            curr_h = sorted([trade_names[j] for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
            
    # --- 3. 整理结果 ---
    res = pd.DataFrame({"nav": nav}, index=df_t.index)
    res['holdings'] = hist
    
    # 截取用户选定的时间段
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
        "raw_acc": acc_df.loc[mask_slice]
    }

# ================= 4. UI 侧边栏：因子调节模块 =================
with st.sidebar:
    st.header("🎛️ 实验参数设置")
    
    with st.expander("1. 基础动量参数", expanded=False):
        rs = st.slider("短期周期 (Fast)", 5, 60, 20)
        rl = st.slider("长期周期 (Slow)", 30, 250, 60)
        rw = st.slider("短期权重", 0, 100, 100) / 100.0
        h = st.number_input("持仓数", 1, 10, 1)
        m = st.number_input("风控均线 (MA)", 5, 120, 20)

    st.markdown("### 2. 新因子调节 (A/B Test)")
    st.info("调整下方参数，对比策略变化")
    
    # RSI 模块
    use_rsi = st.checkbox("启用 RSI 熔断", value=False)
    rsi_limit = st.slider("RSI 上限阈值", 50, 95, 80, 
                          help="当 RSI 超过此数值时，禁止开仓/持仓 (防止追高)")
    
    # 加速度模块
    use_acc = st.checkbox("启用 加速度 过滤", value=False)
    acc_limit = st.slider("加速度 下限阈值", -0.2, 0.1, -0.05, 0.01,
                          help="当 (短期-长期) < 此数值时，禁止开仓 (防止动能衰竭)")

    st.divider()
    with st.expander("📅 时间区间", expanded=True):
        col_d1, col_d2 = st.columns(2)
        start_d = col_d1.date_input("开始", datetime.date.today() - datetime.timedelta(days=365*3))
        end_d = col_d2.date_input("结束", datetime.date.today())

# 打包参数
params = {
    "rs": rs, "rl": rl, "rw": rw, "h": h, "m": m,
    "rsi_period": 14, "rsi_limit": rsi_limit, "acc_limit": acc_limit
}

# ================= 5. 主界面：因子有效性分析 =================
st.title("🧪 动能工厂 - 因子有效性分析实验室")

df = get_clean_data(st.session_state.my_assets, start_d, end_d)

if not df.empty:
    
    # --- 核心：运行两次回测进行对比 ---
    with st.spinner("正在进行 A/B 测试 (基准 vs 新策略)..."):
        # 1. 运行基准策略 (无新因子，所有开关强制 False)
        res_base = run_strategy_engine(df, st.session_state.my_assets, params, start_d, 
                                       use_rsi_filter=False, use_acc_filter=False)
        
        # 2. 运行实验策略 (带用户选定的因子，开关状态由用户决定)
        res_new = run_strategy_engine(df, st.session_state.my_assets, params, start_d, 
                                      use_rsi_filter=use_rsi, use_acc_filter=use_acc)

    if res_base and res_new:
        nav_base = res_base['res']['nav']
        nav_new = res_new['res']['nav']
        
        # --- 1. 效果对比卡片 ---
        st.subheader("📊 实验结果报告")
        
        # 计算核心指标函数
        def calc_metrics(nav_series):
            total_ret = nav_series.iloc[-1] - 1
            mdd = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min()
            daily_rets = nav_series.pct_change().dropna()
            if daily_rets.std() != 0:
                sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252))
            else:
                sharpe = 0
            return total_ret, mdd, sharpe

        ret_b, mdd_b, shp_b = calc_metrics(nav_base)
        ret_n, mdd_n, shp_n = calc_metrics(nav_new)

        c1, c2, c3, c4 = st.columns(4)
        
        # 收益对比
        delta_ret = ret_n - ret_b
        c1.metric("累计收益 (新 vs 旧)", f"{ret_n:.2%}", 
                  delta=f"{delta_ret:.2%}", delta_color="normal")
        
        # 回撤对比 (回撤是负数，如果新回撤(比如-10%) > 旧回撤(比如-20%)，delta是正数，代表改善)
        delta_mdd = mdd_n - mdd_b
        c2.metric("最大回撤", f"{mdd_n:.2%}", 
                  delta=f"{delta_mdd:.2%}", delta_color="inverse")
        
        # 夏普对比
        c3.metric("夏普比率", f"{shp_n:.2f}", 
                  delta=f"{shp_n - shp_b:.2f}")
        
        # 触发统计
        filter_msg = []
        if use_rsi: filter_msg.append(f"RSI拦截 {res_new['stats']['rsi_triggered']} 次")
        if use_acc: filter_msg.append(f"衰竭拦截 {res_new['stats']['acc_triggered']} 次")
        c4.metric("因子拦截统计", " | ".join(filter_msg) if filter_msg else "未启用过滤")

        # --- 2. 净值走势对比 ---
        tab1, tab2 = st.tabs(["📈 净值曲线对比", "🔬 详细信号诊断"])
        
        with tab1:
            fig = go.Figure()
            # 基准线 (灰色虚线)
            fig.add_trace(go.Scatter(x=nav_base.index, y=nav_base, name="原始策略 (基准)", 
                                     line=dict(color='gray', width=2, dash='dot')))
            # 新策略线 (亮色实线)
            fig.add_trace(go.Scatter(x=nav_new.index, y=nav_new, name="优化策略 (当前)", 
                                     line=dict(color='#00ff88', width=3)))
            
            # 判断结果并给出评语
            if ret_n > ret_b and abs(mdd_n) < abs(mdd_b):
                st.success(f"🎉 **正优化！** 引入指标后，收益提升且回撤减小。当前参数有效。")
            elif ret_n < ret_b and abs(mdd_n) < abs(mdd_b):
                st.info(f"🛡️ **防御增强**。收益略降，但安全性（回撤）提高了。适合保守风格。")
            elif ret_n < ret_b:
                st.warning(f"⚠️ **负优化**。过滤条件过严，错过了上涨行情。建议放宽阈值。")
            else:
                st.write("ℹ️ 策略表现基本持平。")

            fig.update_layout(height=500, title="策略净值走势 A/B Test", template="plotly_dark",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("##### 🔍 因子诊断快照 (最新交易日)")
            st.caption("这里展示所有标的在最新一日的状态，帮助你理解为什么它被选入或被剔除。")
            
            # 提取最后一天的快照数据
            last_idx = -1
            snapshot = []
            
            # 获取数据
            r_score = res_new['raw_scores'].iloc[last_idx]
            r_price = res_new['raw_prices'].iloc[last_idx]
            r_rsi = res_new['raw_rsi'].iloc[last_idx]
            r_acc = res_new['raw_acc'].iloc[last_idx]
            
            for name in r_score.index:
                if pd.isna(r_score[name]) or pd.isna(r_price[name]): continue
                
                # 重新判断条件
                base_cond = (r_score[name] > 0) # 动能>0
                rsi_cond = (r_rsi[name] < rsi_limit) # RSI达标
                acc_cond = (r_acc[name] > acc_limit) # 加速度达标
                
                status = "✅ 入选"
                reason = "符合当前策略"
                
                if not base_cond:
                    status = "⚪ 无动能"
                    reason = "动能分<0 或 均线下方"
                
                # 关键修改：区分“未启用”和“已拦截”
                elif not rsi_cond:
                    if use_rsi:
                        status = "⛔ RSI熔断"
                        reason = f"RSI({r_rsi[name]:.1f}) > {rsi_limit}"
                    else:
                        status = "⚠️ 入选(高危)"
                        reason = f"RSI超标({r_rsi[name]:.1f})但未拦截"
                
                elif not acc_cond:
                    if use_acc:
                        status = "⛔ 衰竭熔断"
                        reason = f"加速度({r_acc[name]:.1%}) < {acc_limit}"
                    else:
                        status = "⚠️ 入选(减速)"
                        reason = f"加速度低({r_acc[name]:.1%})但未拦截"
                
                snapshot.append({
                    "标的": name,
                    "动能评分": r_score[name],
                    "RSI": r_rsi[name],
                    "加速度": r_acc[name],
                    "状态": status,
                    "原因": reason
                })
            
            df_snap = pd.DataFrame(snapshot).sort_values("动能评分", ascending=False)
            
            # 表格颜色样式
            def color_status(val):
                if "熔断" in val: return 'color: #ff4444; font-weight: bold' # 红
                if "高危" in val: return 'color: #ffaa00; font-weight: bold' # 橙
                if "入选" in val: return 'color: #00ff88; font-weight: bold' # 绿
                return 'color: gray'

            st.dataframe(
                df_snap.style.format({"动能评分": "{:.2%}", "RSI": "{:.1f}", "加速度": "{:.2%}"})
                .map(color_status, subset=['状态']),
                use_container_width=True,
                height=600
            )

else:
    st.error("📡 无法获取数据，请检查网络连接或尝试缩短时间范围。")
