# ==========================================
# 💡 新增模块：微信自动推送 (Auto Push)
# ==========================================
import requests
import json

def send_wechat_msg(title, content):
    """发送微信推送"""
    token = '你的_PUSHPLUS_TOKEN'  # <--- 请在这里填入你的 Token
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
