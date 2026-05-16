const API_BASE = "http://127.0.0.1:8000";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

let selectedFile = null;
let lastResult = null;
let chatMessages = [];

function fmt(value, suffix = "", digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

function setApiState(ok, text) {
  $("#apiDot").classList.toggle("live", ok);
  $("#apiText").textContent = text;
}

function switchPage(page) {
  $$(".nav-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.page === page));
  $$(".page").forEach((el) => el.classList.toggle("active", el.id === `page-${page}`));
  if (page === "monitor") loadMonitor();
  if (page === "feedback") loadEvents();
  if (page === "rhythm") loadRhythm();
  if (page === "community") loadCommunity();
  if (page === "settings") loadSettings();
}

function metric(label, value, hint = "") {
  return `<article><label>${label}</label><strong>${value}</strong><span>${hint}</span></article>`;
}

function renderPerformance(metrics = {}) {
  const stages = [
    ["ActionCLIP", metrics.actionclip_ms, "动作分类"],
    ["RhythmMamba", metrics.rhythm_ms, "节律建模"],
    ["Bayesian Router", metrics.router_ms, "不确定性路由"],
    ["FastVLM", metrics.vlm_ms, "语义复核"],
    ["Storage", metrics.storage_ms, "数据库写入"],
  ];
  const maxMs = Math.max(...stages.map(([, value]) => Number(value || 0)), 1);
  $("#totalLatency").textContent = fmt(metrics.total_ms, " ms");
  $("#perfBreakdown").innerHTML = stages.map(([name, value, hint]) => {
    const ratio = Math.max(2, Math.min(100, (Number(value || 0) / maxMs) * 100));
    return `
      <div class="perf-row">
        <div>
          <strong>${name}</strong>
          <span>${hint}</span>
        </div>
        <div class="perf-track"><div class="perf-fill" style="width:${ratio}%"></div></div>
        <em>${fmt(value, " ms")}</em>
      </div>
    `;
  }).join("");
}

function renderSummary(result) {
  const ac = result.actionclip || {};
  const router = result.router || {};
  const metrics = result.metrics || {};
  const route = result.vlm_used ? "FastVLM" : "Edge";
  $("#summaryGrid").innerHTML = [
    metric("最终判决", result.final_label || "-", result.source || ""),
    metric("路由", route, (router.route_reasons || []).join(", ") || "no escalation"),
    metric("总延迟", fmt(metrics.total_ms, " ms"), "end to end"),
    metric("显存峰值", fmt(metrics.gpu_mem_peak_mb, " MB", 0), "CUDA peak"),
  ].join("");
  $("#vlmText").textContent = result.vlm_text || "无语义输出。";
  renderPerformance(metrics);
}

function renderTimeline(result) {
  const ac = result.actionclip || {};
  const rhythm = result.rhythm || {};
  const router = result.router || {};
  const metrics = result.metrics || {};
  const rows = [
    ["ActionCLIP", ac.top_label || "-", `confidence ${fmt(ac.confidence, "", 3)}`],
    ["RhythmMamba", `S=${fmt(rhythm.surprise, "", 3)}`, `hour=${rhythm.hour ?? "-"}`],
    ["Bayesian Router", fmt(router.score, "", 3), router.should_route ? "route to VLM" : "pass on edge"],
    ["FastVLM", result.vlm_label || "skipped", fmt(metrics.vlm_ms, " ms")],
  ];
  $("#timeline").innerHTML = rows.map(([name, primary, secondary]) => `
    <div class="timeline-item">
      <div>
        <strong>${name}</strong>
        <span>${primary}</span>
        <small>${secondary}</small>
      </div>
    </div>
  `).join("");
}

function renderProbBars(result) {
  const probs = [...((result.actionclip || {}).probabilities || [])]
    .sort((a, b) => b.probability - a.probability);
  if (!probs.length) {
    $("#probBars").textContent = "暂无概率输出。";
    return;
  }
  $("#probBars").innerHTML = probs.map((item) => {
    const pct = Math.max(0, Math.min(100, item.probability * 100));
    return `
      <div class="bar-row">
        <span>${item.label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <strong>${pct.toFixed(1)}%</strong>
      </div>
    `;
  }).join("");
}

function renderDebug(result) {
  $("#debugJson").textContent = JSON.stringify(result, null, 2);
}

async function runVideo() {
  if (!selectedFile) {
    $("#runState").textContent = "请先选择视频。";
    return;
  }
  $("#runBtn").disabled = true;
  $("#runState").textContent = "正在推理，模型会真实运行，请稍等...";
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    const force = $("#forceVlm").checked;
    const persist = $("#persistResult").checked;
    const result = await api(`/demo/video?force_vlm=${force}&persist=${persist}`, {
      method: "POST",
      body: form,
    });
    lastResult = result;
    if (!result.ok) throw new Error(result.error || "推理失败");
    renderSummary(result);
    renderTimeline(result);
    renderProbBars(result);
    renderDebug(result);
    $("#runState").textContent = `完成：${result.final_label}，frames=${result.frames_used}`;
  } catch (error) {
    $("#runState").textContent = `失败：${error.message}`;
  } finally {
    $("#runBtn").disabled = false;
  }
}

async function loadMonitor() {
  try {
    const status = await api("/status");
    const latest = await api("/latest_event");
    $("#latestEvent").innerHTML = `
      <strong>${latest.title || "未知状态"}</strong>
      <p>${latest.detail || "暂无详情"}</p>
      <small>${latest.time || ""} · camera=${status.camera_online ? "online" : "offline"}</small>
    `;
    const metrics = status.metrics || {};
    const latestMetric = metrics.latest || {};
    $("#metricsGrid").innerHTML = [
      metric("平均延迟", fmt(metrics.avg_total_ms, " ms"), "recent mean"),
      metric("P95 延迟", fmt(metrics.p95_total_ms, " ms"), "tail"),
      metric("VLM 触发率", fmt((metrics.vlm_rate || 0) * 100, "%"), "route load"),
      metric("显存峰值", fmt(latestMetric.gpu_mem_peak_mb, " MB", 0), "last run"),
    ].join("");
  } catch (error) {
    $("#latestEvent").textContent = `无法连接后端：${error.message}`;
  }
}

async function loadEvents() {
  const select = $("#eventSelect");
  try {
    const events = await api("/events?limit=30");
    select.innerHTML = events.map((event) => `
      <option value="${event.id}">#${event.id} · ${event.timestamp} · ${event.raw_label} · conf=${fmt(event.confidence, "", 2)}</option>
    `).join("");
    select.dataset.events = JSON.stringify(events);
  } catch (error) {
    select.innerHTML = "";
    $("#feedbackResult").textContent = `加载失败：${error.message}`;
  }
}

async function generateFeedback() {
  const events = JSON.parse($("#eventSelect").dataset.events || "[]");
  const event = events.find((item) => String(item.id) === $("#eventSelect").value);
  if (!event) return;
  $("#feedbackResult").textContent = "正在调用 LLM...";
  try {
    const result = await api("/generate_feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event }),
    });
    $("#feedbackResult").textContent = result.text || result.error || "无返回。";
  } catch (error) {
    $("#feedbackResult").textContent = `生成失败：${error.message}`;
  }
}

function renderChat() {
  $("#chatLog").innerHTML = chatMessages.map((msg) => `
    <div class="msg ${msg.role === "user" ? "user" : ""}">${msg.content}</div>
  `).join("");
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
}

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chatInput");
  const content = input.value.trim();
  if (!content) return;
  chatMessages.push({ role: "user", content });
  input.value = "";
  renderChat();
  try {
    const result = await api("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chatMessages, limit: 30 }),
    });
    chatMessages.push({ role: "assistant", content: result.text || result.error || "无返回。" });
  } catch (error) {
    chatMessages.push({ role: "assistant", content: `请求失败：${error.message}` });
  }
  renderChat();
}

async function loadRhythm() {
  const box = $("#rhythmChart");
  box.textContent = "加载中...";
  try {
    let rows = await api("/rhythm?limit=96");
    if (!rows.length) {
      const events = await api("/events?limit=96");
      rows = events.reverse().map((event) => ({
        surprise: event.rhythm_surprise ?? event.confidence ?? 0,
      }));
    }
    const values = rows.map((row) => Number(row.surprise || 0));
    const max = Math.max(...values, 0.01);
    box.innerHTML = values.map((value) => `
      <div class="column" title="${value.toFixed(3)}" style="height:${Math.max(2, (value / max) * 100)}%"></div>
    `).join("");
  } catch (error) {
    box.textContent = `加载失败：${error.message}`;
  }
}

async function loadCommunity() {
  const list = $("#communityList");
  list.textContent = "加载中...";
  try {
    const events = await api("/community/high_risk?limit=50");
    if (!events.length) {
      list.textContent = "暂无高风险事件。";
      return;
    }
    list.innerHTML = events.map((event) => `
      <div class="event-card">
        <strong>#${event.id} · ${event.timestamp} · ${event.raw_label}</strong>
        <p>${event.vlm_description || "无描述"}</p>
        <div class="event-actions">
          <button data-action="已处理" data-id="${event.id}">已处理</button>
          <button data-action="标记误报" data-id="${event.id}">标记误报</button>
          <button data-action="请求介入" data-id="${event.id}">请求介入</button>
        </div>
      </div>
    `).join("");
  } catch (error) {
    list.textContent = `加载失败：${error.message}`;
  }
}

async function sendCommunityAction(target) {
  const eventId = Number(target.dataset.id);
  const action = target.dataset.action;
  target.disabled = true;
  try {
    await api("/community/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: eventId, action, operator: "web_demo" }),
    });
    target.textContent = "已记录";
  } catch (error) {
    target.textContent = "失败";
    console.error(error);
  }
}

async function loadSettings() {
  try {
    const cfg = await api("/llm/config");
    $("#baseUrl").value = cfg.base_url || "https://api.openai.com/v1";
    $("#modelName").value = cfg.model || "gpt-4o-mini";
    $("#temperature").value = cfg.temperature ?? 0.3;
    $("#apiKey").placeholder = `当前：${cfg.api_key || "未配置"}`;
  } catch (error) {
    $("#settingsResult").textContent = `加载失败：${error.message}`;
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const payload = {
    base_url: $("#baseUrl").value,
    model: $("#modelName").value,
    temperature: Number($("#temperature").value || 0.3),
  };
  if ($("#apiKey").value) payload.api_key = $("#apiKey").value;
  try {
    const result = await api("/llm/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("#settingsResult").textContent = result.ok ? "已保存。" : (result.error || "保存失败。");
  } catch (error) {
    $("#settingsResult").textContent = `保存失败：${error.message}`;
  }
}

function bindEvents() {
  $$(".nav-item").forEach((btn) => btn.addEventListener("click", () => switchPage(btn.dataset.page)));
  $("#videoFile").addEventListener("change", (event) => {
    selectedFile = event.target.files[0] || null;
    $("#fileName").textContent = selectedFile ? selectedFile.name : "尚未选择文件";
    if (selectedFile) {
      $("#preview").src = URL.createObjectURL(selectedFile);
      $("#preview").hidden = false;
      $("#runState").textContent = "视频已选择，可以开始推理。";
    }
  });
  $("#runBtn").addEventListener("click", runVideo);
  $("#refreshMonitor").addEventListener("click", loadMonitor);
  $("#loadEvents").addEventListener("click", loadEvents);
  $("#generateFeedback").addEventListener("click", generateFeedback);
  $("#chatForm").addEventListener("submit", sendChat);
  $("#clearChat").addEventListener("click", () => {
    chatMessages = [];
    renderChat();
  });
  $("#refreshRhythm").addEventListener("click", loadRhythm);
  $("#loadCommunity").addEventListener("click", loadCommunity);
  $("#communityList").addEventListener("click", (event) => {
    if (event.target.matches("button[data-action]")) sendCommunityAction(event.target);
  });
  $("#settingsForm").addEventListener("submit", saveSettings);
}

async function boot() {
  bindEvents();
  try {
    await api("/status");
    setApiState(true, "API online");
  } catch (error) {
    setApiState(false, "API offline");
  }
  renderTimeline({});
  renderPerformance({});
}

boot();
