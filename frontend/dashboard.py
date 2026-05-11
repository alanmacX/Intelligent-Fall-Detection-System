import streamlit as st
import requests
import time
import os
import sys
import sqlite3
import pandas as pd

# 路径适配
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from components import render_header, render_status_card, render_history_table, render_rhythm_chart

# 数据库路径 (核心：前端直接读库，绕过 Server 限制)
# 假设你的目录结构是 Intelligent_Fall_Detection_System/core/data/guardian.db
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "core", "data", "guardian.db")
API_BASE_URL = "http://127.0.0.1:8000"


# 加载 CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.set_page_config(page_title="Guardian Console", page_icon="🛡️", layout="wide")
load_css(os.path.join(CURRENT_DIR, "styles.css"))

# --- 侧边栏 ---
with st.sidebar:
    st.title("🛡️ 监护控制台")
    mode = st.radio("模式选择", ["实时监控 (iOS Sync)", "数据回溯 (DB Read)", "设置"])

    st.divider()
    # 检查数据库文件是否存在
    if os.path.exists(DB_PATH):
        st.success(f"✅ 数据库已连接")
    else:
        st.error(f"❌ 未找到数据库: {DB_PATH}")

render_header()

# ==========================================
# 🔴 实时监控 (逻辑同 iOS Widget)
# ==========================================
if mode == "实时监控 (iOS Sync)":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎥 实时状态")
        status_box = st.empty()

    with col2:
        st.subheader("📊 节律模拟")
        render_rhythm_chart()

    # 自动刷新逻辑
    try:
        # 像 iOS App 一样调用 /latest_event
        resp = requests.get(f"{API_BASE_URL}/latest_event", timeout=0.5)
        if resp.status_code == 200:
            data = resp.json()
            with status_box:
                render_status_card(data)

            # 弹窗提示
            if data.get("risk_level") == "high":
                st.toast(f"🚨 警报: {data.get('title')}", icon="⚠️")
        else:
            status_box.error("后端返回异常")

    except requests.exceptions.ConnectionError:
        status_box.error("无法连接后端 (Server Offline)")
        st.info("请先运行 python api/server.py")

    time.sleep(1)
    st.rerun()

# ==========================================
# 📜 数据回溯 (直接读 SQLite)
# ==========================================
elif mode == "数据回溯 (DB Read)":
    st.subheader("📜 数据库真实记录")

    if st.button("🔄 刷新数据"):
        st.rerun()

    if os.path.exists(DB_PATH):
        try:
            # 直接连接 SQLite 读取数据，不需要 Server 提供接口
            conn = sqlite3.connect(DB_PATH)
            query = "SELECT timestamp, raw_label, confidence, vlm_description, is_router_active FROM events ORDER BY id DESC LIMIT 50"
            df = pd.read_sql_query(query, conn)
            conn.close()

            render_history_table(df)
        except Exception as e:
            st.error(f"读取数据库失败: {e}")
    else:
        st.warning("找不到数据库文件，请先运行 Server 生成数据。")

# ==========================================
# ⚙️ 设置
# ==========================================
elif mode == "设置":
    st.subheader("⚙️ 参数配置")
    st.info("此界面仅做演示，参数修改需重启 Server")
    st.slider("ActionCLIP 灵敏度", 0.0, 1.0, 0.6)