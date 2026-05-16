import json
import os
from urllib import request, error


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "llm_config.json")


DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in data.items() if v is not None})
    return config


def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    current = load_config()
    current.update(config)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    safe = current.copy()
    safe["api_key"] = mask_key(safe.get("api_key", ""))
    return safe


def mask_key(api_key):
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "****"
    return api_key[:4] + "..." + api_key[-4:]


def public_config():
    config = load_config()
    config["api_key"] = mask_key(config.get("api_key", ""))
    return config


def chat_completion(messages, temperature=None, max_tokens=512):
    config = load_config()
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "LLM API key is not configured"}

    base_url = (config.get("base_url") or DEFAULT_CONFIG["base_url"]).rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": config.get("model") or DEFAULT_CONFIG["model"],
        "messages": messages,
        "temperature": temperature if temperature is not None else float(config.get("temperature", 0.3)),
        "max_tokens": max_tokens,
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return {"ok": True, "text": text, "model": payload["model"], "raw": data}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def generate_event_feedback(event):
    label = event.get("raw_label", "UNKNOWN")
    risk = "high" if label == "FALL" else "low"
    if event.get("rhythm_surprise") and event.get("rhythm_surprise") >= 1.0 and label != "FALL":
        risk = "medium"

    style = {
        "high": "严肃、指令性、高优先级，面向家属，强调立即查看与联系。",
        "medium": "关怀、解释性，面向家属，说明异常但避免制造恐慌。",
        "low": "温和、建议性，面向老人，给出简短健康提醒。",
    }[risk]
    system = (
        "你是智能养老监护系统的语义反馈模块。"
        "你只能基于给定结构化事件生成中文反馈，不要编造医学诊断。"
        f"当前风险等级为{risk}，语气要求：{style}"
    )
    user = (
        "请根据以下事件生成一段不超过80字的反馈，并给出一条行动建议。\n"
        f"{json.dumps(event, ensure_ascii=False)}"
    )
    result = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.25,
        max_tokens=220,
    )
    if result.get("ok"):
        result["risk_level"] = risk
    return result


def answer_health_question(messages, context):
    system = (
        "你是一个智能养老助手。以下是老人近期的行为监测记录，"
        "请基于记录回答家属问题。不得编造未出现的事实；涉及医疗诊断时建议咨询医生。\n\n"
        f"近期记录：\n{context}"
    )
    return chat_completion([{"role": "system", "content": system}] + messages, temperature=0.35, max_tokens=700)
