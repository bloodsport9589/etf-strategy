import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="全球动能工厂-最终版", page_icon="🏭", layout="wide")

# URL 参数持久化功能
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
q = st.query_params

def update_url():
    st.query_params.update({
        "rs": st.session_state.rs, "rl": st.session_state.rl, 
        "rw": st.session_state.rw, "h": st.session_state.h, "m": st.session_state.m
    })

# ================= 2. 侧边栏与参数 =================
with st.sidebar:
    st.header("🎛️ 策略控制台")
    
    # 获取 URL 缓存的参数值
    rs = st.slider("短期ROC", 5, 60, int(q.get("rs", 20)), key="rs", on_change=update_url)
    rl = st.slider("长期ROC", 30, 250, int(q.get("rl", 60)), key="rl", on_change=update_url)
    rw = st.slider("短期权重%", 0, 100, int(q.get("rw", 100)), key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数量", 1, 5, int(q.get("h", 1)), key="h", on_change=update_url)
    m = st.number_input("止损线(MA)", 5, 120, int(q.get("m", 20)), key="m", on_change=update_url)
    
    st.divider() # 这行现在不会报错了
    start_d = st.date_input("回测起点", datetime.date(2022, 1, 1))

# ================= 3. 高效数据获取 =================
@st.cache_data(ttl=3600)
def get_safe_data(start_date):
    assets = {
        "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
        "518880.SS": "黄金ETF", "510300.SS": "沪深300", "^GSPC": "标普500"
    }
    with st.spinner('同步全球行情中...'):
        try:
            data = yf.download(list(assets.keys()), start=start_date, progress=False, timeout=15)
            if data.empty: return pd.DataFrame()
            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
            df.index = df.index.tz_localize(None)
            return df.rename(columns=assets).ffill().dropna(how='all')
        except Exception as e:
            st.error(f"数据加载异常: {e}")
            return pd.DataFrame()

# ================= 4. 回测计算 (极速版) =================
def run_bt(df, rs, rl, rw, h, m):
    # 剔除基准标的
    targets = [c for c in df.columns if c not in ["沪深300", "标普500"]]
    dft = df[targets]
    
    # 动能评分与均线
    score = (dft.pct_change(rs)*rw) + (dft.pct_change(rl)*(1-rw))
    ma = dft.rolling(m).mean()
    rets = dft.pct_change()
    
    nav = np.ones(len(df))
    warm = max(rs, rl, m)
    
    for i in range(warm, len(df)-1):
        s_row = score.values[i]
        mask = (s_row > 0) & (dft.values[i] > ma.values[i])
        if np.any(mask):
            idx = np.where(mask)[0]
            top = idx[np.argsort(s_row[idx])[-h:]]
            nav[i+1] = nav[i] * (1 + rets.values[i+1][top].mean())
        else:
            nav[i+1] = nav[i]
            
    return pd.Series(nav, index=df.index).iloc[warm:]

# ================= 5. 渲染展示 =================
st.title("🏭 全球动能工厂")

df_all = get_safe_data(start_d)

if not df_all.empty:
    nav = run_bt(df_all, rs, rl, rw, h, m)
    
    # 绘图
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略净值", line=dict(color='#00ff88', width=2.5)))
    
    if "标普500" in df_all.columns:
        b = df_all["标普500"].loc[nav.index[0]:]
        fig.add_trace(go.Scatter(x=b.index, y=b/b.iloc[0], name="标普500基准", line=dict(dash='dot', color='gray')))

    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 绩效指标
    c1, c2, c3 = st.columns(3)
    c1.metric("累计收益", f"{nav.iloc[-1]-1:.1%}")
    # 夏普比率
    dr = nav.pct_change().dropna()
    sr = (dr.mean()*252) / (dr.std()*np.sqrt(252)) if len(dr)>0 else 0
    c2.metric("夏普比率", f"{sr:.2f}")
    c3.metric("最大回撤", f"{((nav - nav.cummax())/nav.cummax()).min():.1%}")
else:
    st.info("数据获取中，请稍候... 若长时间无响应请刷新。")
