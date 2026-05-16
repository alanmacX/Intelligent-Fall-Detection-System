import streamlit as st
import requests
import time
import os
import sys
import sqlite3
import pandas as pd
import plotly.express as px

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from components import (
    render_header,
    render_status_card,
    render_history_table,
    render_rhythm_chart,
    render_metrics,
    render_pipeline,
    render_section_title,
)

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "core", "data", "guardian.db")
API_BASE_URL = "http://127.0.0.1:8000"
API_SESSION = requests.Session()
API_SESSION.trust_env = False


def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.set_page_config(page_title="Guardian Console", page_icon="🛡️", layout="wide")
load_css(os.path.join(CURRENT_DIR, "styles.css"))

with st.sidebar:
    st.title("Guardian AI")
    st.caption("Intelligent fall detection workspace")
    mode = st.radio(
        "功能导航",
        ["视频 Demo", "实时监控", "语义反馈", "健康咨询", "节律分析", "社区响应", "数据回溯", "设置"],
    )

    st.divider()
    if os.path.exists(DB_PATH):
        st.success("数据库已连接")
    else:
        st.error(f"未找到数据库: {DB_PATH}")

render_header()

if mode == "实时监控":
    col1, col2 = st.columns([2, 1])

    with col1:
        render_section_title("实时判定", "最新事件、风险等级和语义状态。")
        status_box = st.empty()
        metrics_box = st.empty()

    with col2:
        render_section_title("节律趋势", "从历史运行记录中观察节律惊奇度。")
        rhythm_box = st.empty()

    try:
        resp = API_SESSION.get(f"{API_BASE_URL}/latest_event", timeout=0.5)
        if resp.status_code == 200:
            data = resp.json()
            with status_box:
                render_status_card(data)
            if data.get("risk_level") == "high":
                st.toast(f"警报: {data.get('title')}")
        else:
            status_box.error("后端返回异常")

        metrics_resp = API_SESSION.get(f"{API_BASE_URL}/metrics/summary", timeout=0.5)
        if metrics_resp.status_code == 200:
            with metrics_box:
                render_metrics(metrics_resp.json())

        rhythm_resp = API_SESSION.get(f"{API_BASE_URL}/rhythm?limit=96", timeout=0.5)
        if rhythm_resp.status_code == 200:
            with rhythm_box:
                render_rhythm_chart(rhythm_resp.json())

    except requests.exceptions.ConnectionError:
        status_box.error("无法连接后端 (Server Offline)")
        st.info("请先运行 python api/server.py")

    time.sleep(1)
    st.rerun()

elif mode == "视频 Demo":
    render_section_title("单视频真实推理", "上传视频后查看模型判决、路由原因、阶段延迟和 CUDA 显存峰值。")
    control_col, preview_col = st.columns([0.9, 1.1], gap="large")
    with control_col:
        uploaded = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi", "mkv"])
        force_vlm = st.toggle("强制 FastVLM 复核", value=False)
        persist = st.toggle("结果写入数据库", value=True)
        run_clicked = st.button("开始推理", type="primary", use_container_width=True)
    with preview_col:
        if uploaded is not None:
            st.video(uploaded)
        else:
            st.markdown(
                "<div class='empty-preview'>等待视频输入。上传后这里会显示预览，并在推理完成后展示真实判决链路。</div>",
                unsafe_allow_html=True,
            )

    if uploaded is not None:
        if run_clicked:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "video/mp4")}
            params = {"force_vlm": str(force_vlm).lower(), "persist": str(persist).lower()}
            with st.spinner("正在运行 ActionCLIP + RhythmMamba + Router + FastVLM..."):
                try:
                    resp = API_SESSION.post(f"{API_BASE_URL}/demo/video", files=files, params=params, timeout=300)
                except requests.exceptions.ConnectionError:
                    st.error("无法连接后端，请先运行 python api/server.py")
                    resp = None

            if resp is not None:
                if resp.status_code != 200:
                    st.error(f"后端返回异常: {resp.status_code}")
                    st.code(resp.text)
                else:
                    result = resp.json()
                    if not result.get("ok"):
                        st.error(result.get("error", "推理失败"))
                    else:
                        render_pipeline(result)

                        with st.expander("完整 Debug JSON"):
                            st.json(result)

elif mode == "语义反馈":
    st.subheader("📝 分级语义反馈")
    try:
        events_resp = API_SESSION.get(f"{API_BASE_URL}/events?limit=20", timeout=2)
        events = events_resp.json() if events_resp.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        events = []
        st.error("无法连接后端")

    if not events:
        st.info("暂无事件。先在“视频 Demo”里上传一个视频并写入数据库。")
    else:
        labels = [f"#{e.get('id')} | {e.get('timestamp')} | {e.get('raw_label')} | conf={e.get('confidence')}" for e in events]
        selected = st.selectbox("选择事件", range(len(events)), format_func=lambda i: labels[i])
        st.json(events[selected])
        if st.button("生成语义反馈", type="primary"):
            with st.spinner("正在调用 LLM..."):
                resp = API_SESSION.post(f"{API_BASE_URL}/generate_feedback", json={"event": events[selected]}, timeout=90)
            data = resp.json()
            if data.get("ok"):
                st.success(f"风险等级: {data.get('risk_level')}")
                st.write(data.get("text"))
                with st.expander("LLM Debug"):
                    st.json({k: v for k, v in data.items() if k != "raw"})
            else:
                st.error(data.get("error", "生成失败"))

elif mode == "健康咨询":
    st.subheader("💬 健康咨询 Chatbot")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("问问老人最近的行为、节律异常或跌倒风险...")
    if user_msg:
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("正在结合历史事件调用 LLM..."):
                resp = API_SESSION.post(
                    f"{API_BASE_URL}/chat",
                    json={"messages": st.session_state.chat_messages, "limit": 30},
                    timeout=120,
                )
                data = resp.json()
            if data.get("ok"):
                answer = data.get("text", "")
                st.write(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                with st.expander("使用的事件上下文"):
                    st.text(data.get("context", ""))
            else:
                st.error(data.get("error", "聊天失败"))

    if st.button("清空对话"):
        st.session_state.chat_messages = []
        st.rerun()

elif mode == "节律分析":
    st.subheader("📈 节律惊奇度分析")
    try:
        rhythm = API_SESSION.get(f"{API_BASE_URL}/rhythm?limit=672", timeout=3).json()
        events = API_SESSION.get(f"{API_BASE_URL}/events?limit=200", timeout=3).json()
    except requests.exceptions.ConnectionError:
        rhythm, events = [], []
        st.error("无法连接后端")

    if rhythm:
        df = pd.DataFrame(rhythm)
        fig = px.line(df, x="timestamp", y="surprise", title="RhythmMamba Surprise")
        fall_events = [e for e in events if e.get("raw_label") == "FALL"]
        for e in fall_events:
            fig.add_vline(x=e.get("timestamp"), line_color="red", line_width=2)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(50), use_container_width=True)
    elif events:
        df = pd.DataFrame(events)
        if "rhythm_surprise" in df.columns and df["rhythm_surprise"].notna().any():
            y_col = "rhythm_surprise"
        else:
            y_col = "confidence"
            st.info("暂无 rhythm_surprise，使用 confidence 作为 fallback 展示。")
        fig = px.line(df, x="timestamp", y=y_col, title=f"{y_col} fallback")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无节律或事件数据。")

elif mode == "社区响应":
    st.subheader("🏘️ 社区响应模拟")
    try:
        events = API_SESSION.get(f"{API_BASE_URL}/community/high_risk?limit=50", timeout=3).json()
    except requests.exceptions.ConnectionError:
        events = []
        st.error("无法连接后端")

    if not events:
        st.info("暂无高风险事件。")
    for event in events:
        with st.container():
            st.markdown(f"**事件 #{event.get('id')} | {event.get('timestamp')} | {event.get('raw_label')}**")
            st.caption(event.get("vlm_description") or "无描述")
            cols = st.columns(3)
            for action, col in zip(["已处理", "标记误报", "请求介入"], cols):
                if col.button(action, key=f"{action}-{event.get('id')}"):
                    resp = API_SESSION.post(
                        f"{API_BASE_URL}/community/action",
                        json={"event_id": event.get("id"), "action": action, "operator": "dashboard"},
                        timeout=5,
                    )
                    data = resp.json()
                    if data.get("ok"):
                        st.success("反馈已记录，样本将进入难例池。")
                    else:
                        st.error(data.get("error", "记录失败"))
            st.divider()

# ==========================================
# ==========================================
elif mode == "数据回溯":
    st.subheader("📜 数据库真实记录")

    if st.button("🔄 刷新数据"):
        st.rerun()

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT timestamp, raw_label, confidence, router_score, router_uncertainty,
                       rhythm_surprise, entropy, margin, vlm_description, is_router_active
                FROM events ORDER BY id DESC LIMIT 50
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            render_history_table(df)
        except Exception as e:
            st.error(f"读取数据库失败: {e}")
    else:
        st.warning("找不到数据库文件，请先运行 Server 生成数据。")

# ==========================================
# ==========================================
elif mode == "设置":
    st.subheader("⚙️ API 配置")
    try:
        cfg = API_SESSION.get(f"{API_BASE_URL}/llm/config", timeout=3).json()
    except requests.exceptions.ConnectionError:
        cfg = {"api_key": "", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "temperature": 0.3}
        st.warning("后端未连接，保存会失败。")

    with st.form("llm_config_form"):
        api_key = st.text_input("API Key", value="", type="password", placeholder=f"当前: {cfg.get('api_key') or '未配置'}")
        base_url = st.text_input("Base URL", value=cfg.get("base_url", "https://api.openai.com/v1"))
        model = st.text_input("Model", value=cfg.get("model", "gpt-4o-mini"))
        temperature = st.slider("Temperature", 0.0, 1.0, float(cfg.get("temperature", 0.3)), 0.05)
        submitted = st.form_submit_button("保存 API 配置", type="primary")

    if submitted:
        payload = {"base_url": base_url, "model": model, "temperature": temperature}
        if api_key:
            payload["api_key"] = api_key
        try:
            resp = API_SESSION.post(f"{API_BASE_URL}/llm/config", json=payload, timeout=5)
            data = resp.json()
            if data.get("ok"):
                st.success("API 配置已保存。")
                st.json(data.get("config"))
            else:
                st.error(data.get("error", "保存失败"))
        except requests.exceptions.ConnectionError:
            st.error("无法连接后端")

    st.caption("兼容 OpenAI Chat Completions 协议；DeepSeek 可填 https://api.deepseek.com 和 deepseek-chat。")
