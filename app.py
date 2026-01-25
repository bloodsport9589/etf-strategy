import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-旗舰版", page_icon="🏭", layout="wide")

# 初始化策略参数
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = int(st.query_params.get(key, val))

# 初始化品种池
DEFAULT_ASSETS = {
    "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
    "510180.SS": "上证180", "159915.SZ": "创业板指", "518880.SS": "黄金ETF",
    "512400.SS": "有色ETF", "159981.SZ": "能源ETF", "588050.SS": "科创50",
    "501018.SS": "南方原油",
}
if 'my_assets' not in st.session_state:
    st.session_state.my_assets = DEFAULT_ASSETS.copy()

def update_url():
    st.query_params.update({k: st.session_state[k] for k in DEFAULTS.keys()})

# ================= 2. 侧边栏：品种管理逻辑修复 =================
with st.sidebar:
    st.header("🎛️ 策略控制")
    
    with st.expander("📝 品种管理", expanded=True):
        st.markdown("**添加新标的**")
        # 使用单独的 key 并避免直接在 button 判断中使用 nc/nn
        new_code = st.text_input("代码 (如 AAPL 或 513100.SS)", key="input_code")
        new_name = st.text_input("名称 (如 苹果 或 纳指ETF)", key="input_name")
        
        if st.button("➕ 确认添加", width="stretch"):
            if new_code and new_name:
                # 更新 session_state 中的字典
                st.session_state.my_assets[new_code] = new_name
                st.toast(f"已添加: {new_name}")
                st.rerun() # 强制刷新以加载新数据
            else:
                st.error("请填写完整的代码和名称")
        
        st.divider()
        st.markdown("**当前品种池**")
        # 遍历删除逻辑
        assets_to_delete = None
        for code, name in list(st.session_state.my_assets.items()):
            cols = st.columns([3, 1])
            cols[0].write(f"{name}\n`{code}`")
            if cols[1].button("❌", key=f"del_{code}"):
                assets_to_delete = code
        
        if assets_to_delete:
            del st.session_state.my_assets[assets_to_delete]
            st.rerun()

        if st.button("🔄 重置为默认品种", width="stretch"):
            st.session_state.my_assets = DEFAULT_ASSETS.copy()
            st.rerun()
            
    st.divider()
    st.subheader("参数设置")
    rs = st.slider("短期ROC (天)", 5, 60, value=st.session_state.rs, key="rs", on_change=update_url)
    rl = st.slider("长期ROC (天)", 30, 250, value=st.session_state.rl, key="rl", on_change=update_url)
    rw = st.slider("短期权重 (%)", 0, 100, value=st.session_state.rw, key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 10, value=st.session_state.h, key="h", on_change=update_url)
    m = st.number_input("止损均线 (MA)", 5, 120, value=st.session_state.m, key="m", on_change=update_url)
    start_d = st.date_input("回测开始", datetime.date(2020, 1, 1))

# ================= 3. 数据与回测引擎 =================
@st.cache_data(ttl=3600)
def get_data(assets_dict, start_date):
    benchmarks = {"510300.SS": "沪深300", "^GSPC": "标普500"}
    targets = {**assets_dict, **benchmarks}
    try:
        data = yf.download(list(targets.keys()), start=start_date, progress=False, timeout=30)
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        else:
            df = data
        df.index = df.index.tz_localize(None)
        return df.rename(columns=targets).ffill().dropna(how='all')
    except: return pd.DataFrame()

@st.cache_data
def run_backtest(df_all, assets, rs, rl, rw, h, m):
    trade_names = [n for n in assets.values() if n in df_all.columns]
    if not trade_names: return None, None, None, None
    
    df_t = df_all[trade_names]
    scores = (df_t.pct_change(rs) * rw) + (df_t.pct_change(rl) * (1-rw))
    ma = df_t.rolling(m).mean()
    rets = df_t.pct_change()
    
    warm_up = max(rs, rl, m)
    nav = np.ones(len(df_t))
    holdings_history = [[] for _ in range(len(df_t))] 

    score_vals, price_vals, ma_vals, ret_vals = scores.values, df_t.values, ma.values, rets.values

    for i in range(warm_up, len(df_t) - 1):
        s_row = score_vals[i]
        mask = (s_row > 0) & (price_vals[i] > ma_vals[i])
        day_pnl = 0.0
        current_holdings = []
        if np.any(mask):
            idx = np.where(mask)[0]
            top_idx = idx[np.argsort(s_row[idx])[-h:]]
            day_pnl = np.nanmean(ret_vals[i+1][top_idx])
            current_holdings = [trade_names[j] for j in top_idx]
        nav[i+1] = nav[i] * (1 + day_pnl)
        holdings_history
