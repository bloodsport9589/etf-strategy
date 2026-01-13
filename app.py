import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="全球动能工厂 (自定义版)", page_icon="🏭", layout="wide")

# ================= 初始默认标的池 =================
DEFAULT_ASSETS = {
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

# ================= 会话状态初始化 (用于存储动态标的) =================
if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

# ================= 侧边栏：控制台 =================
st.sidebar.header("🎛️ 策略控制台")

# --- 模块 0: 标的管理 (新增功能) ---
with st.sidebar.expander("📝 标的管理 (自定义)", expanded=False):
    st.write("自主添加或删除回测品种")
    
    # 添加功能
    c1, c2 = st.columns([1, 1])
    new_code = c1.text_input("代码 (6位)", max_chars=6, placeholder="513330")
    new_name = c2.text_input("名称", placeholder="恒生互联")
    
    if st.button("➕ 添加品种"):
        if len(new_code) == 6 and len(new_name) > 0:
            st.session_state.my_assets[new_code] = new_name
            st.success(f"已添加: {new_name}")
            st.rerun() # 立即刷新
        else:
            st.error("请输入正确的代码和名称")
            
    # 删除功能
    st.divider()
    current_list = [f"{code} : {name}" for code, name in st.session_state.my_assets.items()]
    del_targets = st.multiselect("选择要删除的品种", current_list)
    
    if st.button("🗑️ 删除选中"):
        for item in del_targets:
            code = item.split(" : ")[0]
            if code in st.session_state.my_assets:
                del st.session_state.my_assets[code]
        st.success("删除成功")
        st.rerun()
        
    # 重置功能
    if st.button("🔄 重置为默认列表"):
        st.session_state.my_assets = DEFAULT_ASSETS.copy()
        st.rerun()

# --- 模块 1: 参数设置 ---
st.sidebar.subheader("1. 策略参数")
ROC_SHORT = st.sidebar.slider("短期 ROC (天)", 5, 60, 20)
ROC_LONG = st.sidebar.slider("长期 ROC (天)", 30, 250, 60)
ROC_WEIGHT = st.sidebar.slider("短期权重 (%)", 0, 100, 100) / 100.0

HOLD_COUNT = st.sidebar.number_input("持仓数量", min_value=1, max_value=10, value=1)
MA_EXIT = st.sidebar.number_input("止损均线 (MA)", min_value=5, max_value=120, value=20)
BACKTEST_START = st.sidebar.date_input("回测开始日期", datetime.date(2020, 1, 1))

# ================= 核心计算逻辑 =================

# 注意：为了让数据随标的变化而更新，这里去掉了 @st.cache_data 
# 或者必须把 assets 字典作为参数传入以触发缓存更新。这里为了从简，直接传入 keys tuple
@st.cache_data(ttl=3600) 
def get_historical_data(start_date, asset_keys_tuple):
    """获取数据 (传入 asset_keys_tuple 是为了让缓存感知到标的列表的变化)"""
    combined_df = pd.DataFrame()
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")
    
    # 动态获取当前 session 中的 assets
    current_assets = st.session_state.my_assets
    targets = {**current_assets, **BENCHMARKS}
    
    progress = st.empty()
    total = len(targets)
    
    for i, (code, name) in enumerate(targets.items()):
        progress.text(f"正在加载 ({i+1}/{total}): {name}...")
        try:
            # 尝试获取数据
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date, adjust="qfq")
            if not df.empty:
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
    # 使用当前动态标的列表
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
    if start_idx >= len(df_trade): return None, None, None
    
    curve = [1.0]
    dates = [df_trade.index[start_idx]]
    
    ret_daily = df_trade.pct_change()
    factor_analysis_data = [] 

    for i in range(start_idx, len(df_trade) - 1):
        scores = score_df.iloc[i]
        prices = df_trade.iloc[i]
        mas = ma_df.iloc[i]
        
        # 交易逻辑
        valid = scores[(scores > 0) & (prices > mas)]
        
        targets = []
        if not valid.empty:
            targets = valid.sort_values(ascending=False).head(HOLD_COUNT).index.tolist()
            
        day_pnl = 0.0
        if targets:
            w = 1.0 / HOLD_COUNT 
            # 防止某个标的数据缺失
            valid_targets = [t for t in targets if t in ret_daily.columns]
            if valid_targets:
                rets = ret_daily.iloc[i+1][valid_targets]
                day_pnl = rets.sum() * w
        
        curve.append(curve[-1] * (1 + day_pnl))
        dates.append(df_trade.index[i+1])
        
        # 因子数据收集
        daily_rank = scores.rank(ascending=False, method='first') 
        next_day_ret = ret_daily.iloc[i+1]
        
        for asset in scores.index:
            r = daily_rank.get(asset)
            ret = next_day_ret.get(asset)
            if pd.notnull(r) and pd.notnull(ret):
                factor_analysis_data.append({"Rank": int(r), "Return": ret})

    return pd.Series(curve, index=dates), pd.DataFrame(factor_analysis_data)

# ================= 主界面 =================

st.title("🏭 全球动能工厂 (自定义版)")

# 为了缓存机制正常工作，我们将字典的 keys 转为 tuple 传入
# 这样当 session_state.my_assets 变化时，函数参数变化，触发重新拉取数据
asset_keys = tuple(sorted(st.session_state.my_assets.keys()))
df_all = get_historical_data(BACKTEST_START, asset_keys)

if not df_all.empty:
    score_df, ma_df, df_trade = calculate_factors(df_all, ROC_SHORT, ROC_LONG, ROC_WEIGHT)
    nav, factor_data = run_backtest(df_trade, score_df, ma_df)
    
    if nav is not None:
        
        # --- Part 1: 今日信号 ---
        st.divider()
        st.header("💡 今日实盘信号")
        
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
                "现价": p,
                "止损线": m,
                "状态": "✅ 持有" if is_buy else "❌ 空仓"
            })
            
        df_rank = pd.DataFrame(rank_data).sort_values("综合动能", ascending=False).reset_index(drop=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("📢 建议操作")
            buys = df_rank[df_rank['状态'] == "✅ 持有"].head(HOLD_COUNT)
            if buys.empty:
                st.error("🛑 空仓 (持有现金)")
            else:
                st.success(f"✅ 买入 Top {HOLD_COUNT}")
                for _, row in buys.iterrows():
                    st.write(f"**{row['名称']}**")
        
        with c2:
            st.subheader("📊 实时排行")
            d_show = df_rank.copy()
            d_show['综合动能'] = d_show['综合动能'].apply(lambda x: f"{x*100:.2f}%")
            d_show['止损线'] = d_show['止损线'].apply(lambda x: f"{x:.3f}")
            def color_status(v):
                return f'color: {"#00ff88" if "✅" in v else "#ff4444"}; font-weight: bold'
            st.dataframe(d_show.style.applymap(color_status, subset=['状态']), use_container_width=True)

        st.divider()
        
        # --- Part 2: 详细分析 ---
        tab1, tab2 = st.tabs(["📈 策略表现", "🔬 因子体检"])
        
        with tab1:
            # 基准
            start_dt = nav.index[0]
            b_nasdaq = df_all.get("纳指ETF")
            b_hs300 = df_all.get("沪深300")
            
            # 绘图
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav.index, y=nav, name='策略净值', line=dict(color='#00ff88', width=3)))
            
            if b_nasdaq is not None:
                b_nasdaq = b_nasdaq.loc[start_dt:] / b_nasdaq.loc[start_dt:].iloc[0]
                fig.add_trace(go.Scatter(x=b_nasdaq.index, y=b_nasdaq, name='纳指ETF', line=dict(color='#3366ff', width=1.5)))
                
            if b_hs300 is not None:
                b_hs300 = b_hs300.loc[start_dt:] / b_hs300.loc[start_dt:].iloc[0]
                fig.add_trace(go.Scatter(x=b_hs300.index, y=b_hs300, name='沪深300', line=dict(color='#ff3333', width=1.5, dash='dot')))

            # 业绩KPI
            total_ret = (nav.iloc[-1] - 1) * 100
            nasdaq_ret = (b_nasdaq.iloc[-1] - 1) * 100 if b_nasdaq is not None else 0
            days = (nav.index[-1] - nav.index[0]).days
            cagr = (nav.iloc[-1] ** (365/days) - 1) * 100 if days > 0 else 0
            dd = ((nav - nav.cummax()) / nav.cummax()).min() * 100
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益", f"{total_ret:.1f}%", delta=f"{total_ret - nasdaq_ret:.1f}%")
            k2.metric("年化", f"{cagr:.1f}%")
            k3.metric("最大回撤", f"{dd:.1f}%")
            k4.metric("持仓数", f"{HOLD_COUNT}")

            fig.update_layout(template="plotly_dark", title="净值对比", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            if not factor_data.empty:
                rank_perf = factor_data.groupby("Rank")["Return"].mean() * 100
                fig_bar = px.bar(
                    x=rank_perf.index, y=rank_perf.values,
                    title="分层回测 (次日平均涨幅)",
                    color=rank_perf.values, color_continuous_scale="RdYlGn"
                )
                fig_bar.update_layout(template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
                
                diff = rank_perf.get(1, 0) - rank_perf.iloc[-1]
                st.metric("多空收益差", f"{diff:.3f}%", help="Top1收益 - Bottom1收益")
            else:
                st.info("暂无数据")

    else:
        st.error("请检查回测参数或数据源")
else:
    st.warning("数据加载中或失败...")
# ==========================================
# 💡 新增模块：微信自动推送 (Auto Push)
# ==========================================
import requests
import json

def send_wechat_msg(title, content):
    """发送微信推送"""
    token = '235cb751b98d4b8b917d523332e56517'  # <--- 请在这里填入你的 Token
    url = 'http://www.pushplus.plus/send'
    data = {
        "token": token,
        "title": title,
        "content": content,
        "template": "html"
    }
    try:
        requests.post(url, json=data)
    except:
        pass

# 侧边栏开关
st.sidebar.divider()
enable_push = st.sidebar.checkbox("开启每日微信推送", value=False)

if enable_push:
    # 检查是否到了推送时间 (比如每天 15:00 收盘后，或者 09:00 开盘前)
    # Streamlit 是被动触发的，你需要保持网页开启，或者使用 GitHub Actions 定时运行
    # 这里演示手动点击触发，或者你每次打开网页时自动触发
    
    # 获取今日建议数据
    latest_scores = score_df.iloc[-1]
    latest_prices = df_trade.iloc[-1]
    latest_mas = ma_df.iloc[-1]
    
    # 生成消息内容
    msg_title = f"【量化日报】{datetime.datetime.now().strftime('%Y-%m-%d')}"
    msg_content = "<h3>今日操作建议：</h3><ul>"
    
    rank_data = []
    for name in latest_scores.index:
        s = latest_scores.get(name, -99)
        p = latest_prices.get(name, 0)
        m = latest_mas.get(name, 0)
        is_buy = (s > 0) and (p > m)
        
        status_icon = "✅" if is_buy else "❌"
        # 只推送前 N 名
        rank_data.append((name, s, is_buy))
        
    # 排序
    rank_data.sort(key=lambda x: x[1], reverse=True)
    top_n = rank_data[:HOLD_COUNT]
    
    has_buy = False
    for name, score, is_buy in top_n:
        if is_buy:
            msg_content += f"<li style='color:green'><b>买入/持有：{name}</b> (动能 {score*100:.1f}%)</li>"
            has_buy = True
        else:
            msg_content += f"<li style='color:red'>空仓观察：{name} (虽排名高但走弱)</li>"
            
    if not has_buy:
        msg_content += "<li><b>🛑 建议全额空仓/现金</b></li>"
        
    msg_content += "</ul><br><a href='https://你的Streamlit网址.streamlit.app'>点击查看详情</a>"
    
    if st.button("📤 手动发送今日推送到微信"):
        send_wechat_msg(msg_title, msg_content)
        st.toast("✅ 推送已发送！请查看微信")
