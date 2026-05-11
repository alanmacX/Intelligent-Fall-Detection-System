import streamlit as st
import pandas as pd
import plotly.express as px


def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>🛡️ 智能监护系统 (iOS 同步版)</h1>
            <p>实时数据流看板 | 端云协同架构</p>
        </div>
    """, unsafe_allow_html=True)


def render_status_card(data):
    """
    渲染与 iOS App 逻辑一致的状态卡片
    """
    if not data:
        st.warning("等待服务连接...")
        return

    # 获取 iOS App 同款字段
    title = data.get("title", "未知状态")  # e.g., "⚠️ 跌倒报警"
    risk = data.get("risk_level", "low")  # e.g., "high"
    time_str = data.get("time", "")
    source = data.get("source", "System")  # e.g., "【云端双重确认】"
    detail = data.get("detail", "")  # e.g., "FastVLM 的描述"
    suggestion = data.get("suggestion", "")

    # 样式映射
    css_class = f"risk-{risk}"

    st.markdown(f"""
        <div class="status-card {css_class}">
            <div style="font-size: 1rem; opacity: 0.8;">{time_str}</div>
            <div style="font-size: 3rem; font-weight: bold; margin: 10px 0;">{title}</div>
            <div style="font-size: 1.2rem; background: rgba(0,0,0,0.1); padding: 5px; border-radius: 5px;">
                💡 {suggestion}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 详情区
    st.markdown(f"""
        <div class="detail-text">
            <div><span class="source-tag">{source}</span> <strong>智能分析：</strong></div>
            <div style="margin-top: 8px;">{detail}</div>
        </div>
    """, unsafe_allow_html=True)


def render_history_table(df):
    """
    渲染历史记录 (直接读取数据库的数据)
    """
    if df.empty:
        st.info("数据库暂无记录")
        return

    # 映射数据库字段名到中文显示
    # 数据库字段: id, timestamp, raw_label, confidence, vlm_description, is_router_active
    display_df = df.rename(columns={
        "timestamp": "时间",
        "raw_label": "原始标签",
        "confidence": "置信度",
        "vlm_description": "语义描述",
        "is_router_active": "VLM介入"
    })

    # 格式化置信度
    if "置信度" in display_df.columns:
        display_df["置信度"] = display_df["置信度"].apply(lambda x: f"{x:.2f}")

    # 格式化 VLM 状态
    if "VLM介入" in display_df.columns:
        display_df["VLM介入"] = display_df["VLM介入"].apply(lambda x: "✅ 是" if x == 1 else "No")

    # 隐藏 ID 列，按时间倒序
    if "id" in display_df.columns:
        display_df = display_df.drop(columns=["id"])

    st.dataframe(display_df, use_container_width=True, height=400)


def render_rhythm_chart():
    # 模拟节律图表 (保持不变)
    hours = list(range(24))
    activity = [0, 0, 0, 0, 0, 1, 5, 8, 7, 4, 5, 6, 7, 2, 3, 6, 8, 9, 6, 4, 2, 1, 0, 0]
    df = pd.DataFrame({"Hour": hours, "Activity": activity})
    fig = px.bar(df, x='Hour', y='Activity', title="今日活动趋势", height=300)
    st.plotly_chart(fig, use_container_width=True)