import pandas as pd
import plotly.express as px
import streamlit as st


def format_num(value, suffix="", digits=1):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def render_header():
    st.markdown(
        """
        <div class="hero-shell">
            <div>
                <div class="eyebrow">Guardian AI · Fall Detection</div>
                <h1>智能跌倒检测推理台</h1>
                <p>单视频真实推理、节律异常、路由复核、语义响应与运行性能统一观测。</p>
            </div>
            <div class="hero-status">
                <span class="status-dot"></span>
                <span>Local inference</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, subtitle=None):
    caption = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{title}</h2>
            {caption}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(items):
    if not items:
        return
    cols = st.columns(min(len(items), 3))
    for idx, (label, value, hint) in enumerate(items):
        with cols[idx % len(cols)]:
            with st.container(border=True):
                st.caption(label)
                st.metric(label="", value=value, label_visibility="collapsed")
                if hint:
                    st.caption(hint)


def render_status_card(data):
    if not data:
        st.warning("等待服务连接...")
        return

    title = data.get("title", "未知状态")
    risk = data.get("risk_level", "low")
    time_str = data.get("time", "")
    source = data.get("source", "System")
    detail = data.get("detail", "")
    suggestion = data.get("suggestion", "")
    risk_text = {"low": "Stable", "medium": "Watch", "high": "Alert"}.get(risk, "Stable")

    st.markdown(
        f"""
        <div class="status-panel risk-{risk}">
            <div class="panel-topline">
                <span>{risk_text}</span>
                <span>{time_str}</span>
            </div>
            <div class="status-main">{title}</div>
            <div class="status-sub">{suggestion}</div>
        </div>
        <div class="analysis-panel">
            <div class="analysis-source">{source}</div>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_table(df):
    if df.empty:
        st.info("数据库暂无记录")
        return

    display_df = df.rename(
        columns={
            "timestamp": "时间",
            "raw_label": "原始标签",
            "confidence": "置信度",
            "vlm_description": "语义描述",
            "is_router_active": "VLM介入",
            "router_score": "Router",
            "router_uncertainty": "不确定性",
            "rhythm_surprise": "节律惊奇度",
            "entropy": "熵",
            "margin": "边界",
        }
    )

    if "置信度" in display_df.columns:
        display_df["置信度"] = display_df["置信度"].apply(lambda x: format_num(x, digits=2))

    if "VLM介入" in display_df.columns:
        display_df["VLM介入"] = display_df["VLM介入"].apply(lambda x: "是" if x == 1 else "否")

    if "id" in display_df.columns:
        display_df = display_df.drop(columns=["id"])

    st.dataframe(display_df, use_container_width=True, height=430, hide_index=True)


def render_rhythm_chart(rows):
    if not rows:
        st.info("等待节律数据...")
        return
    df = pd.DataFrame(rows)
    fig = px.line(df, x="timestamp", y="surprise", height=320)
    fig.update_traces(line_color="#007aff", line_width=2.5)
    fig.update_layout(
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1d1d1f"),
        xaxis_title=None,
        yaxis_title="Surprise",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metrics(metrics):
    if not metrics:
        st.info("等待推理指标...")
        return
    latest = metrics.get("latest", {}) or {}
    render_metric_row(
        [
            ("平均延迟", format_num(metrics.get("avg_total_ms"), " ms"), f"最近 {metrics.get('count', 0)} 次"),
            ("P95 延迟", format_num(metrics.get("p95_total_ms"), " ms"), "尾部耗时"),
            ("VLM 触发率", format_num((metrics.get("vlm_rate") or 0) * 100, "%"), "复核负载"),
            ("显存峰值", format_num(latest.get("gpu_mem_peak_mb"), " MB", 0), "最近一次 CUDA peak"),
        ]
    )


def render_performance_breakdown(metrics):
    if not metrics:
        st.info("暂无性能数据")
        return

    stages = [
        ("ActionCLIP", metrics.get("actionclip_ms"), "视觉动作分类"),
        ("Rhythm", metrics.get("rhythm_ms"), "节律惊奇度"),
        ("Router", metrics.get("router_ms"), "贝叶斯路由"),
        ("FastVLM", metrics.get("vlm_ms"), "语义复核"),
        ("Storage", metrics.get("storage_ms"), "持久化"),
    ]
    max_ms = max([float(v or 0) for _, v, _ in stages] + [1.0])
    with st.container(border=True):
        head_left, head_right = st.columns([1, 1])
        head_left.markdown("**性能分解**")
        head_right.caption(f"total {format_num(metrics.get('total_ms'), ' ms')}")
        for name, value, hint in stages:
            progress = min(1.0, max(0.0, float(value or 0) / max_ms))
            left, right = st.columns([1, 4])
            left.markdown(f"**{name}**")
            left.caption(hint)
            right.progress(progress, text=format_num(value, " ms"))


def render_pipeline(result):
    ac = result.get("actionclip", {})
    rhythm = result.get("rhythm", {})
    router = result.get("router", {})
    metrics = result.get("metrics", {})
    route_state = "FastVLM" if result.get("vlm_used") else "Edge"

    render_metric_row(
        [
            ("最终判决", result.get("final_label", "-"), result.get("source", "")),
            ("路由", route_state, ", ".join(router.get("route_reasons", [])) or "no escalation"),
            ("置信度", format_num(ac.get("confidence"), digits=3), ac.get("top_label", "")),
            ("总延迟", format_num(metrics.get("total_ms"), " ms"), "end to end"),
            ("显存峰值", format_num(metrics.get("gpu_mem_peak_mb"), " MB", 0), "CUDA peak"),
            ("节律惊奇度", format_num(rhythm.get("surprise"), digits=3), rhythm.get("model", "")),
        ]
    )
    render_performance_breakdown(metrics)

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        render_section_title("判决链路", "ActionCLIP output, rhythm context, Bayesian route.")
        timeline = [
            ("ActionCLIP", ac.get("top_label", "-"), format_num(ac.get("confidence"), digits=3)),
            ("RhythmMamba", f"S={format_num(rhythm.get('surprise'), digits=3)}", f"hour={rhythm.get('hour', '-')}"),
            ("Bayesian Router", format_num(router.get("score"), digits=3), "route" if router.get("should_route") else "pass"),
            ("FastVLM", result.get("vlm_label") or "skipped", format_num(metrics.get("vlm_ms"), " ms")),
        ]
        with st.container(border=True):
            for name, primary, secondary in timeline:
                row_left, row_right = st.columns([1, 2])
                row_left.markdown(f"**{name}**")
                row_right.write(primary)
                row_right.caption(secondary)

    with right:
        render_section_title("语义输出", "FastVLM text or edge explanation.")
        st.info(result.get("vlm_text") or "无语义输出。")

    probs = pd.DataFrame(ac.get("probabilities", []))
    if not probs.empty:
        fig = px.bar(
            probs.sort_values("probability", ascending=True),
            x="probability",
            y="label",
            orientation="h",
            height=390,
        )
        fig.update_traces(marker_color="#007aff")
        fig.update_layout(
            margin=dict(l=8, r=8, t=10, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1d1d1f"),
            xaxis_title="Probability",
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
