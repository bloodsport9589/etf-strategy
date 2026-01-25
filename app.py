import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np

# ================= 1. 基础配置 =================
st.set_page_config(page_title="极速量化", page_icon="⚡", layout="wide")

# URL 参数持久化
DEFAULTS = {"rs": 20, "rl": 60, "rw": 100, "h": 1, "m": 20}
q = st.query_params

def update_url():
    st.query_params.update({"rs":st.session_state.rs, "rl":st.session_state.rl, "rw":st.session_state.rw, "h":st.session_state.h, "m":st.session_state.m})

# ================= 2. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 配置")
    rs = st.slider("短期ROC", 5, 60, int(q.get("rs", 20)), key="rs", on_change=update_url)
    rl = st.slider("长期ROC", 30, 250, int(q.get("rl", 60)), key="rl", on_change=update_url)
    rw = st.slider("短期权重%", 0, 100, int(q.get("rw", 100)), key="rw", on_change=update_url) / 100.0
    h = st.number_input("持仓数", 1, 5, int(q.get("h", 1)), key="h", on_change=update_url)
    m = st.number_input("止损线", 5, 120, int(q.get("m", 20)), key="m", on_change=update_url)
    start_d = st.date_input("开始日期", datetime.date(2022, 1, 1)) # 默认缩短时间以提速

# ================= 3. 数据引擎 (增加稳定性) =================
@st.cache_data(ttl=3600)
def get_data_v2(start_date):
    assets = {
        "513100.SS": "纳指ETF", "513520.SS": "日经ETF", "513180.SS": "恒生科技",
        "510300.SS": "沪深300", "^GSPC": "标普500", "518880.SS": "黄金ETF"
    }
    codes = list(assets.keys())
    
    # 使用 st.spinner 确保用户知道在干嘛
    with st.spinner('正在从全球服务器同步行情...'):
        try:
            # 增加 timeout 参数防止卡死
            data = yf.download(codes, start=start_date, progress=False, timeout=20)
            if data.empty: return pd.DataFrame()
            
            df = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
            df.index = df.index.tz_localize(None)
            return df.rename(columns=assets).ffill().dropna(how='all')
        except Exception as e:
            st.error(f"连接超时或失败: {e}")
            return pd.DataFrame()

# ================= 4. 回测逻辑 (向量化) =================
def fast_bt(df, rs, rl, rw, h, m):
    # 仅选择交易标的 (排除基准)
    trades = [c for c in df.columns if c not in ["沪深300", "标普500"]]
    dft = df[trades]
    
    # 指标计算
    score = (dft.pct_change(rs)*rw) + (dft.pct_change(rl)*(1-rw))
    ma = dft.rolling(m).mean()
    rets = dft.pct_change()
    
    # 模拟
    nav = np.ones(len(df))
    warm = max(rs, rl, m)
    
    for i in range(warm, len(df)-1):
        # 选股
        s_row = score.values[i]
        mask = (s_row > 0) & (dft.values[i] > ma.values[i])
        if np.any(mask):
            idx = np.where(mask)[0]
            # 选动能最高的 h 个
            top = idx[np.argsort(s_row[idx])[-h:]]
            nav[i+1] = nav[i] * (1 + rets.values[i+1][top].mean())
        else:
            nav[i+1] = nav[i] # 空仓
            
    return pd.Series(nav, index=df.index).iloc[warm:]

# ================= 5. 渲染 =================
st.title("⚡ 极速动能分析")

df = get_data_v2(start_d)

if not df.empty:
    nav = fast_bt(df, rs, rl, rw, h, m)
    
    # 简易绘图以提速
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略", line=dict(color='#00ff88', width=2)))
    
    # 仅增加标普作为对比
    if "标普500" in df.columns:
        b = df["标普500"].loc[nav.index[0]:]
        fig.add_trace(go.Scatter(x=b.index, y=b/b.iloc[0], name="标普500", line=dict(dash='dot', color='gray')))

    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 指标
    c1, c2, c3 = st.columns(3)
    c1.metric("累计收益", f"{nav.iloc[-1]-1:.1%}")
    # 夏普简化版
    dr = nav.pct_change().dropna()
    sr = (dr.mean()*252) / (dr.std()*np.sqrt(252)) if len(dr)>0 else 0
    c2.metric("夏普比率", f"{sr:.2f}")
    mdd = ((nav - nav.cummax())/nav.cummax()).min()
    c3.metric("最大回撤", f"{mdd:.1%}")
else:
    st.info("💡 正在等待数据响应... 如果长时间没反应，请尝试刷新页面。")
