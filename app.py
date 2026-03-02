import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import requests
import time

# ================= 1. 基础配置 & 默认参数 =================
st.set_page_config(page_title="全球动能工厂-实盘追踪版", page_icon="🏭", layout="wide")

# 采用经过真实数据洗礼的稳健参数作为默认值 (20日/60日，MA20风控)
DEFAULTS = {
    "rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20,
    "rsi_period": 14, "rsi_limit": 80, "acc_limit": -0.05 
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 资产池字典
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "豆粕ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油" 
}
# 沪深300作为交易日历基准
BENCHMARKS = {"510300.SS": "沪深300"}

if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# 初始化实盘交易记录表 (后台自动计算现金流)
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame({
        "Date": [datetime.date(2026, 2, 13)],
        "Action": ["买入"],
        "Asset": ["日经ETF"], 
        "Price": [1.000],      
        "Volume": [10000.0]
    })

# ================= 2. 核心算法与数据引擎 =================

def calculate_rsi_series(series, period=14):
    """计算 RSI 指标"""
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
    import yfinance as yf
    import requests
    import pandas as pd
    import time

    combined_df = pd.DataFrame()
    error_logs = []

    progress_bar = st.progress(0)
    total = len(assets_dict)

    for i, (ticker, name) in enumerate(assets_dict.items()):
        progress_bar.progress((i + 1) / total)
        series = None
        
        # 🟢 1. 南方原油专属通道 (场内价格 + 场外净值双核兜底)
        if "501018" in ticker:
            # 策略 A：添加防盗链 Referer，尝试获取场内交易价
            try:
                headers_east = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Referer": "http://quote.eastmoney.com/"  # 突破 Connection aborted 的通关密语！
                }
                url = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.501018&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53&klt=101&fqt=2&end=20500101&lmt=1000"
                res = requests.get(url, headers=headers_east, timeout=5).json()
                klines = res['data']['klines']
                dates = [k.split(',')[0] for k in klines]
                closes = [float(k.split(',')[2]) for k in klines]
                series = pd.Series(closes, index=pd.to_datetime(dates), name=name)
            except Exception as e1:
                # 策略 B：如果场内价格仍被墙，直接拉取天天基金的官方净值！(海外绝对可用)
                try:
                    headers_fund = {
                        "User-Agent": "Mozilla/5.0",
                        "Referer": "http://fundf10.eastmoney.com/"
                    }
                    # pageSize=500 代表拉取近 500 个交易日的净值
                    url_fund = "http://api.fund.eastmoney.com/f10/lsjz?fundCode=501018&pageIndex=1&pageSize=500"
                    res_fund = requests.get(url_fund, headers=headers_fund, timeout=5).json()
                    
                    if res_fund.get('Data') and res_fund['Data'].get('LSJZList'):
                        jz_list = res_fund['Data']['LSJZList']
                        dates = [item['FSRQ'] for item in jz_list]
                        closes = [float(item['DWJZ']) for item in jz_list if item['DWJZ']]
                        series = pd.Series(closes, index=pd.to_datetime(dates), name=name)
                        series = series.sort_index() # 净值是倒序的，必须排正
                    else:
                        raise ValueError("天天基金接口返回为空")
                except Exception as e2:
                    error_logs.append(f"南方原油彻底失败: 场内({e1}) / 净值({e2})")
        
        # 🔵 2. 其它 ETF 使用 YFinance 原生拉取
        else:
            try:
                tk = yf.Ticker(ticker)
                df_yf = tk.history(period="3y") 
                
                if not df_yf.empty:
                    series = df_yf['Close']
                    series.name = name
            except Exception as e:
                error_logs.append(f"{name} 抓取失败: {e}")

        # 🟡 3. 数据纯净合并 (剥离时区，对齐日期)
        if series is not None and not series.empty:
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            series.index = pd.to_datetime(series.index).normalize()
            
            # 去除重复日期的数据
            series = series[~series.index.duplicated(keep='last')]
            
            if combined_df.empty:
                combined_df = pd.DataFrame({name: series})
            else:
                combined_df = combined_df.join(series, how='outer')
                
        time.sleep(0.1)

    progress_bar.empty()

    if error_logs:
        st.error("⚠️ 数据诊断日志：\n" + "\n".join(error_logs))

    if combined_df.empty:
        return combined_df

    # 🚀 强制正序排列，解决负值问题！
    combined_df = combined_df.sort_index(ascending=True)
    combined_df = combined_df.ffill().dropna(how='all')
    
    start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=365) 
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    
    mask = (combined_df.index >= start_dt) & (combined_df.index <= end_dt)
    return combined_df.loc[mask]

# ================= 3. 策略核心逻辑 =================
def run_strategy_engine(df_all, assets, params, user_start_date):
    """动能评分与信号过滤引擎"""
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
    
    is_tradeable = (df_t.diff() != 0).fillna(True) 
    warm_up = max(rs, rl, m, rsi_p)
    nav = np.ones(len(df_t))
    hist = [[] for _ in range(len(df_t))]
    
    s_vals, p_vals, m_vals = scores.values, df_t.values, ma.values
    r_vals, rsi_vals, acc_vals = rets.values, rsi_df.values, acc_df.values
    t_vals = is_tradeable.values

    for i in range(warm_up, len(df_t) - 1):
        valid_data = np.isfinite(s_vals[i]) & np.isfinite(p_vals[i]) & np.isfinite(m_vals[i])
        
        # 核心过滤条件
        base_signal = (s_vals[i] > 0) & (p_vals[i] > m_vals[i]) & t_vals[i]
        pass_rsi = (rsi_vals[i] < rsi_limit)
        pass_acc = (acc_vals[i] > acc_limit)
        
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
        "res": res, "raw_scores": scores.loc[mask_slice], "raw_prices": df_t.loc[mask_slice],
        "raw_rsi": rsi_df.loc[mask_slice], "raw_acc": acc_df.loc[mask_slice],
        "raw_ma": ma.loc[mask_slice], "raw_tradeable": is_tradeable.loc[mask_slice]
    }

# ================= 4. 实盘净值计算引擎 =================
def calculate_real_portfolio(df_prices, trade_history, start_date_str="2026-02-13", initial_nav=1.0):
    """根据交易记录核算真实净值与现金流"""
    if df_prices.empty or trade_history.empty:
        return None, None
        
    start_dt = pd.to_datetime(start_date_str)
    df_p = df_prices.loc[df_prices.index >= start_dt].copy()
    if df_p.empty: return None, None

    positions = {name: 0.0 for name in DEFAULT_ASSETS.values()}
    cash = 0.0
    daily_total_value = []
    
    trades = trade_history.copy()
    trades['Date'] = pd.to_datetime(trades['Date']).dt.date
    
    def calc_cash_flow(row):
        try:
            val = float(row['Price']) * float(row['Volume'])
            return -val if row['Action'] == "买入" else val
        except: return 0.0
    trades['Cash_Flow'] = trades.apply(calc_cash_flow, axis=1)
    
    trades = trades.sort_values("Date")
    trade_idx, num_trades = 0, len(trades)
    
    for current_date in df_p.index:
        current_date_date = current_date.date()
        
        while trade_idx < num_trades:
            trade_date = trades.iloc[trade_idx]['Date']
            if trade_date <= current_date_date:
                trade = trades.iloc[trade_idx]
                matched_name = next((name for name in positions.keys() if name in str(trade['Asset'])), None)
                        
                if matched_name:
                    if trade['Action'] == "买入":
                        positions[matched_name] += float(trade['Volume'])
                        cash += float(trade['Cash_Flow'])
                    elif trade['Action'] == "卖出":
                        positions[matched_name] -= float(trade['Volume'])
                        cash += float(trade['Cash_Flow'])
                trade_idx += 1
            else:
                break

        market_value = sum(vol * float(df_p.loc[current_date, asset]) for asset, vol in positions.items() if vol > 0 and asset in df_p.columns)
        total_assets = cash + market_value
        daily_total_value.append(total_assets)
        
    res_df = pd.DataFrame({"Total_Assets": daily_total_value}, index=df_p.index)
    initial_assets = res_df['Total_Assets'].iloc[0]
    res_df['Real_NAV'] = 0.0 if initial_assets == 0 else (res_df['Total_Assets'] / initial_assets) * initial_nav
        
    final_state = {"cash": cash, "market_value": market_value, "positions": positions}
    return res_df, final_state

# ================= 5. UI 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 策略参数微调")
    with st.expander("当前动能与风控参数", expanded=True):
        rs = st.slider("短期周期 (Fast)", 5, 60, st.session_state['rs'])
        rl = st.slider("长期周期 (Slow)", 30, 250, st.session_state['rl'])
        rw = st.slider("短期权重", 0, 100, int(st.session_state['rw']*100)) / 100.0
        h = st.number_input("持仓数量", 1, 10, st.session_state['h'])
        m = st.number_input("均线防守 (MA)", 5, 120, st.session_state['m'])
        rsi_limit = st.slider("RSI 熔断上限", 50, 95, st.session_state['rsi_limit'])
        acc_limit = st.slider("加速度过滤下限", -0.3, 0.1, st.session_state['acc_limit'], 0.01)

    st.divider()
    col_d1, col_d2 = st.columns(2)
    start_d = col_d1.date_input("回测开始", datetime.date.today() - datetime.timedelta(days=365*3))
    end_d = col_d2.date_input("回测结束", datetime.date.today())

params = {
    "rs": rs, "rl": rl, "rw": rw, "h": h, "m": m,
    "rsi_period": 14, "rsi_limit": rsi_limit, "acc_limit": acc_limit
}

# ================= 6. 主界面 =================
st.title("🧪 动能工厂 - 全球资产实盘版 🚀")

df = get_clean_data(st.session_state.my_assets, start_d, end_d)

if not df.empty:
    missing_assets = [name for name in st.session_state.my_assets.values() if name not in df.columns]
    if missing_assets:
        st.warning(f"⚠️ **数据诊断**：以下标的缺失：{', '.join(missing_assets)}")

if df.empty:
    st.error("❌ 数据获取失败。请检查网络。")
else:
    tab1, tab2, tab3 = st.tabs(["💰 个人实盘资金曲线", "📈 策略每日诊断播报", "⚙️ 历史参数回测基准"])
    
    # ---------------- 页面 1：实盘资金曲线 ----------------
    with tab1:
        st.markdown("### 📝 手动实盘调仓记录表")
        st.info("💡 输入单价和数量后，系统会自动在后台计算扣除/增加的账户现金。基准起始 1.0000。")
        
        asset_options = list(st.session_state.my_assets.values())
        edited_df = st.data_editor(
            st.session_state.trade_history, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("交易日", required=True),
                "Action": st.column_config.SelectboxColumn("买卖动作", options=["买入", "卖出"], required=True),
                "Asset": st.column_config.SelectboxColumn("标的", options=asset_options, required=True),
                "Price": st.column_config.NumberColumn("成交单价", format="%.3f", required=True),
                "Volume": st.column_config.NumberColumn("成交份数", step=100, required=True),
            }
        )
        st.session_state.trade_history = edited_df
        
        if st.button("🔄 根据真实行情合并计算净值"):
            with st.spinner("正在核算..."):
                real_nav_df, final_state = calculate_real_portfolio(df, st.session_state.trade_history)
                
            if real_nav_df is not None:
                current_nav = real_nav_df['Real_NAV'].iloc[-1]
                st.metric(label="当前实盘绝对净值", value=f"{current_nav:.4f}", delta=f"{(current_nav-1.0):.2%}")
                
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("账户剩余现金", f"¥ {final_state['cash']:,.2f}")
                c_b.metric("当前持仓市值", f"¥ {final_state['market_value']:,.2f}")
                c_c.metric("账户总资产", f"¥ {(final_state['cash'] + final_state['market_value']):,.2f}")
                
                fig_real = go.Figure()
                fig_real.add_trace(go.Scatter(x=real_nav_df.index, y=real_nav_df['Real_NAV'], name="实盘净值", line=dict(color='#ff00ff', width=3)))
                
                trade_dates = pd.to_datetime(st.session_state.trade_history['Date']).dt.date
                for dt in trade_dates:
                    try:
                        valid_dt = real_nav_df.index[real_nav_df.index.date >= dt][0]
                        nav_val = real_nav_df.loc[valid_dt, 'Real_NAV']
                        fig_real.add_annotation(x=valid_dt, y=nav_val, text="🔄 调仓", showarrow=True, arrowhead=1, ax=0, ay=-40)
                    except: pass
                    
                fig_real.update_layout(height=400, template="plotly_dark", title="📈 账户绝对净值走势 (基准 1.00)")
                st.plotly_chart(fig_real, use_container_width=True)

    # ---------------- 页面 2：策略诊断 ----------------
    with tab2:
        with st.spinner("正在根据最新行情诊断信号..."):
            res_new = run_strategy_engine(df, st.session_state.my_assets, params, start_d)
            
        if res_new is not None and not res_new['raw_scores'].empty:
            last_date = res_new['raw_scores'].index[-1]
            st.markdown(f"### 🔎 {last_date.strftime('%Y-%m-%d')} 收盘后真实数据诊断")
            
            last_idx = -1
            r_score, r_price, r_ma = res_new['raw_scores'].iloc[last_idx], res_new['raw_prices'].iloc[last_idx], res_new['raw_ma'].iloc[last_idx]
            r_rsi, r_acc, r_trad = res_new['raw_rsi'].iloc[last_idx], res_new['raw_acc'].iloc[last_idx], res_new['raw_tradeable'].iloc[last_idx]
            real_holdings = res_new['res']['holdings'].iloc[last_idx]
            
            snapshot = []
            for name in r_score.index:
                if name not in r_price.index or pd.isna(r_score[name]): continue
                
                is_above_ma, is_pos_score = r_price[name] > r_ma[name], r_score[name] > 0
                rsi_ok, acc_ok = r_rsi[name] < rsi_limit, r_acc[name] > acc_limit
                
                if not r_trad[name]: status, reason = "🚫 停牌", "无波动"
                elif name in real_holdings: status, reason = "✅ 建议买入/持有", "综合排名第一且满足风控"
                else:
                    if not is_pos_score: status, reason = "⚪ 落选", "动能为负"
                    elif not is_above_ma: status, reason = "⚪ 落选", f"跌破MA{m}均线"
                    elif not rsi_ok: status, reason = "⛔ 熔断", f"RSI({r_rsi[name]:.1f}) 极度超买"
                    elif not acc_ok: status, reason = "⛔ 剔除", f"加速度({r_acc[name]:.2f}) 衰竭"
                    else: status, reason = "⚠️ 备选排队", "各项健康但动能稍逊"

                snapshot.append({
                    "标的": name, f"短动能({rs}日)": r_score[name], "加速度": r_acc[name],
                    "RSI(14)": r_rsi[name], "当前状态": status, "诊断原因": reason
                })
            
            if snapshot:
                df_snap = pd.DataFrame(snapshot).sort_values(f"短动能({rs}日)", ascending=False)
                def color_row(val):
                    if "持有" in val: return 'color: #00ff88; font-weight: bold; background-color: rgba(0,255,136,0.1)'
                    if "熔断" in val or "剔除" in val: return 'color: #ff4444; font-weight: bold'
                    if "备选" in val: return 'color: #ffcc00'
                    return 'color: gray'

                st.dataframe(
                    df_snap.style.format({f"短动能({rs}日)": "{:.2%}", "加速度": "{:.2f}", "RSI(14)": "{:.1f}"})
                    .map(color_row, subset=['当前状态']), use_container_width=True, height=400
                )
                
                if real_holdings:
                    st.success(f"🎯 **策略指令：当前应重点持仓 👉 {', '.join(real_holdings)}**")
                else:
                    st.warning("🛑 **策略指令：所有资产均未通过安全风控，强制保持 👉 空仓 (持有现金)**")

    # ---------------- 页面 3：历史回测 ----------------
    with tab3:
        if res_new is not None:
            nav_new = res_new['res']['nav']
            def calc_metrics(nav):
                if len(nav) < 2: return 0, 0, 0 
                ret = nav.iloc[-1] - 1
                mdd = ((nav - nav.cummax()) / nav.cummax()).min()
                dr = nav.pct_change().dropna()
                shp = (dr.mean()*252)/(dr.std()*np.sqrt(252)) if dr.std()!=0 else 0
                return ret, mdd, shp
                
            rn, mn, sn = calc_metrics(nav_new)
            
            st.markdown("### 📊 参数历史基准测试")
            c1, c2, c3 = st.columns(3)
            c1.metric("区间累计收益", f"{rn:.2%}")
            c2.metric("区间最大回撤", f"{mn:.2%}")
            c3.metric("年化夏普比率", f"{sn:.2f}")
            
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(x=nav_new.index, y=nav_new, name="纯策略净值", line=dict(color='#00ff88', width=2)))
            fig_backtest.update_layout(height=450, template="plotly_dark", title="📈 理论策略资金曲线 (不含实盘滑点)", hovermode="x unified")
            st.plotly_chart(fig_backtest, use_container_width=True)
