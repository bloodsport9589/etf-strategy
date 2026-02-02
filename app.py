import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-拐点侦测版", page_icon="🏭", layout="wide")

# 初始化参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20, "rsi_period": 14}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        try:
            p_val = st.query_params.get(key, str(val))
            st.session_state[key] = int(p_val)
        except:
            st.session_state[key] = val

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

def update_url():
    params = {k: st.session_state[k] for k in DEFAULTS.keys() if k in st.session_state}
    st.query_params.update(params)

# ================= 2. 辅助函数：计算 RSI =================
def calculate_rsi_series(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan) # 避免除以0
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # 填充初始值

# ================= 3. 数据引擎 =================
@st.cache_data(ttl=3600)
def get_clean_data(assets_dict, start_date, end_date):
    targets = {**assets_dict, **BENCHMARKS}
    fetch_start = start_date - timedelta(days=365)
    fetch_end = end_date + timedelta(days=1)
    
    try:
        data = yf.download(list(targets.keys()), start=fetch_start, end=fetch_end, progress=False, group_by='ticker')
        if data.empty: return pd.DataFrame()

        clean_data = pd.DataFrame()
        
        for ticker in targets.keys():
            try:
                if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                    col_data = data[ticker]
                elif ticker in data.columns:
                    col_data = data[[ticker]]
                else:
                    continue

                if 'Adj Close' in col_data.columns:
                    s = col_data['Adj Close']
                elif 'Close' in col_data.columns:
                    s = col_data['Close']
                else:
                    s = col_data.iloc[:, 0]
                
                if s.dropna().empty: continue 
                clean_data[ticker] = s

            except Exception:
                continue 

        if clean_data.empty: return pd.DataFrame()
        
        rename_map = {k: v for k, v in targets.items() if k in clean_data.columns}
        clean_data = clean_data.rename(columns=rename_map)
        clean_data.index = clean_data.index.tz_localize(None)
        clean_data = clean_data.ffill().dropna(how='all')
        
        return clean_data
    except Exception:
        return pd.DataFrame()

# ================= 4. 增强版回测引擎 =================
@st.cache_data
def run_enhanced_backtest(df_all, assets, rs, rl, rw, h, m, user_start_date, rsi_p):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None, 0, None, None
    
    df_t = df_all[trade_names]
    
    # --- 1. 动量计算 ---
    mom_short = df_t.pct_change(rs)
    mom_long = df_t.pct_change(rl)
    scores = (mom_short * rw) + (mom_long * (1-rw))
    
    # --- 2. 拐点侦测指标 ---
    # (A) 加速度：短期动能 - 长期动能 (如果为负，说明涨势变慢，即使动能分为正)
    # 为了量纲统一，我们简单用 mom_short - mom_long
    acceleration = mom_short - mom_long
    
    # (B) RSI 指标
    rsi_df = df_t.apply(lambda x: calculate_rsi_series(x, rsi_p))

    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m, rsi_p)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    trade_count = 0

    s_vals = scores.values
    p_vals = df_t.values
    m_vals = ma.values
    r_vals = rets.values
    
    # 回测循环
    for i in range(warm_up, len(df_t) - 1):
        valid_data_mask = np.isfinite(s_vals[i]) & np.isfinite(p_vals[i]) & np.isfinite(m_vals[i])
        signal_mask = (s_vals[i] > 0) & (p_vals[i] > m_vals[i])
        final_mask = valid_data_mask & signal_mask
        
        day_pnl = 0.0
        curr_h = []
        
        if np.any(final_mask):
            idx = np.where(final_mask)[0]
            # 选分最高的 Top H
            top_idx = idx[np.argsort(s_vals[i][idx])[-h:]]
            
            day_pnl = np.nanmean(r_vals[i+1][top_idx])
            if np.isnan(day_pnl): day_pnl = 0.0
            curr_h = sorted([trade_names[j] for j in top_idx])
        
        nav[i+1] = nav[i] * (1 + day_pnl)
        hist[i+1] = curr_h
        if hist[i+1] != hist[i]: trade_count += 1
            
    # 数据截取
    full_res = pd.DataFrame({"nav": nav, "holdings": hist}, index=df_t.index)
    mask_slice = full_res.index >= pd.to_datetime(user_start_date)
    res_sliced = full_res.loc[mask_slice].copy()
    
    if res_sliced.empty: return None, None, None, None, 0, None, None
        
    res_sliced['nav'] = res_sliced['nav'] / res_sliced['nav'].iloc[0]
    
    scores_sliced = scores.loc[mask_slice]
    ma_sliced = ma.loc[mask_slice]
    df_t_sliced = df_t.loc[mask_slice]
    # 返回额外的指标供分析
    acc_sliced = acceleration.loc[mask_slice]
    rsi_sliced = rsi_df.loc[mask_slice]
    
    return res_sliced, scores_sliced, ma_sliced, df_t_sliced, trade_count, acc_sliced, rsi_sliced

# ================= 5. UI 渲染 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    with st.expander("📅 回测区间", expanded=True):
        col_d1, col_d2 = st.columns(2)
        start_d = col_d1.date_input("开始", datetime.date.today() - datetime.timedelta(days=365*2))
        end_d = col_d2.date_input("结束", datetime.date.today())

    with st.expander("⚙️ 核心参数", expanded=True):
        rs = st.slider("短期周期 (Fast)", 5, 60, key="rs", on_change=update_url)
        rl = st.slider("长期周期 (Slow)", 30, 250, key="rl", on_change=update_url)
        rw = st.slider("短期权重", 0, 100, key="rw", on_change=update_url) / 100.0
        h = st.number_input("持仓数", 1, 10, key="h", on_change=update_url)
        m = st.number_input("均线 (MA)", 5, 120, key="m", on_change=update_url)
        rsi_p = st.number_input("RSI 周期", 5, 30, 14, key="rsi_period", on_change=update_url)

st.title("🏭 全球动能工厂 - 拐点侦测版")
st.caption("引入加速度分析与 RSI 过热检测，辅助判断趋势末端")

df = get_clean_data(st.session_state.my_assets, start_d, end_d)

if not df.empty:
    # 这里的解包增加了 acc (加速度) 和 rsi (相对强弱)
    bt_res = run_enhanced_backtest(df, st.session_state.my_assets, rs, rl, rw, h, m, start_d, rsi_p)
    res_df, score_df, ma_df, df_trade, t_count, acc_df, rsi_df = bt_res if bt_res[0] is not None else (None,)*7
    
    if res_df is not None:
        nav = res_df['nav']
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累计收益", f"{nav.iloc[-1]-1:.2%}")
        k2.metric("最大回撤", f"{((nav - nav.cummax()) / nav.cummax()).min():.2%}")
        k3.metric("交易次数", t_count)
        k4.metric("当前策略状态", "运行中" if nav.iloc[-1] > 0 else "停止")

        # --- 绘图 ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略净值", line=dict(color='#00ff88', width=3)))
        for b_name in BENCHMARKS.values():
            if b_name in df.columns:
                b_val = df[b_name].loc[nav.index]
                fig.add_trace(go.Scatter(x=b_val.index, y=b_val/b_val.iloc[0], name=b_name, line=dict(dash='dot'), opacity=0.5))
        st.plotly_chart(fig, use_container_width=True)

        # --- 核心修改：智能信号分析榜单 ---
        st.divider()
        st.subheader("🔍 动量拐点雷达")
        st.info("💡 解读：动能评分高+加速中=最强；动能高+减速=可能见顶；RSI>80=极度危险")

        if not score_df.empty:
            last_idx = -1
            l_score = score_df.iloc[last_idx]
            l_price = df_trade.iloc[last_idx]
            l_ma = ma_df.iloc[last_idx]
            l_acc = acc_df.iloc[last_idx]  # 加速度
            l_rsi = rsi_df.iloc[last_idx]  # RSI

            ranks = []
            for name in l_score.index:
                if pd.isna(l_score[name]) or pd.isna(l_price[name]): continue
                
                # 基础信号
                is_hold = (l_score[name] > 0) and (l_price[name] > l_ma[name])
                
                # 拐点分析逻辑
                status_emoji = ""
                status_text = ""
                
                # 1. 速度判断
                if l_acc[name] > 0.02: # 短期比长期涨得快很多
                    acc_status = "🚀 加速中"
                elif l_acc[name] < -0.01: # 短期明显跑输长期
                    acc_status = "🐢 动力衰竭"
                else:
                    acc_status = "➡️ 匀速"
                
                # 2. 过热判断
                rsi_val = l_rsi[name]
                if rsi_val > 80:
                    rsi_status = "🔥 严重超买"
                elif rsi_val > 70:
                    rsi_status = "⚠️ 偏高"
                elif rsi_val < 30:
                    rsi_status = "❄️ 超卖"
                else:
                    rsi_status = "✅ 正常"

                # 综合建议
                if is_hold:
                    if rsi_val > 80:
                        advice = "建议止盈 (过热)"
                        color = "#ff4444" # 红
                    elif l_acc[name] < -0.05:
                        advice = "注意风险 (减速)"
                        color = "#ffaa00" # 橙
                    else:
                        advice = "持有"
                        color = "#00ff88" # 绿
                else:
                    advice = "空仓"
                    color = "#777777" # 灰

                ranks.append({
                    "标的": name,
                    "动能评分": l_score[name],
                    "加速度": l_acc[name],
                    "RSI(14)": rsi_val,
                    "趋势状态": f"{acc_status} | {rsi_status}",
                    "决策建议": advice,
                    "_color": color 
                })

            df_rank = pd.DataFrame(ranks).sort_values("动能评分", ascending=False)
            
            # 使用 Pandas Styler 进行着色
            def color_advice(val):
                if "止盈" in val: return 'color: red; font-weight: bold'
                if "风险" in val: return 'color: orange; font-weight: bold'
                if "持有" in val: return 'color: #00ff88; font-weight: bold'
                return 'color: gray'

            st.dataframe(
                df_rank.style.format({
                    "动能评分": "{:.2%}", "加速度": "{:.2%}", "RSI(14)": "{:.1f}"
                })
                .map(color_advice, subset=['决策建议'])
                .bar(subset=['动能评分'], color='#3366cc', vmin=-0.2, vmax=0.2),
                use_container_width=True,
                height=500
            )

    else:
        st.warning("数据不足")
else:
    st.error("无法获取数据")
