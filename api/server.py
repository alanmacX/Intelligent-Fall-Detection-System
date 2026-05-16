import sys
import os
import tempfile

# ==========================================
# ==========================================
current_file_path = os.path.abspath(__file__)
api_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(api_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time

try:
    from core.engine import GuardianEngine
    from api.llm_client import (
        public_config,
        save_config,
        generate_event_feedback,
        answer_health_question,
    )
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"🔍 调试: sys.path[0] = {sys.path[0]}")
    sys.exit(1)

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = GuardianEngine()
    yield
    if engine:
        engine.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://0.0.0.0:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackPayload(BaseModel):
    event_id: int
    feedback_type: str
    note: str = ""


class LLMConfigPayload(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3


class GenerateFeedbackPayload(BaseModel):
    event: dict | None = None
    event_id: int | None = None


class ChatPayload(BaseModel):
    messages: list[dict]
    limit: int = 30


class CommunityActionPayload(BaseModel):
    event_id: int
    action: str
    operator: str = "demo_operator"


def load_video_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


@app.get("/")
def root():
    return {"status": "running", "service": "Guardian AI"}


@app.get("/status")
def status():
    is_cam_online = bool(engine and engine.perception and engine.perception.is_online())
    latest = None
    metrics = {}
    if engine and engine.db:
        events = engine.db.get_recent_events(limit=1)
        latest = events[0] if events else None
        metrics = engine.db.get_metrics_summary(limit=100)
    return {
        "service": "Guardian AI",
        "camera_online": is_cam_online,
        "latest_event": latest,
        "metrics": metrics,
    }


@app.get("/latest_event")
def get_latest_event():
    """
    Return the latest monitoring payload and camera status.
    """
    is_cam_online = False
    if engine and engine.perception:
        is_cam_online = engine.perception.is_online()

    response = {
        "title": "系统启动中...",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera_online": is_cam_online,
        "risk_level": "low",
        "source": "System",
        "detail": "等待数据流接入...",
        "suggestion": "正在初始化..."
    }

    if engine and engine.db:
        try:
            events = engine.db.get_recent_events(limit=1)
            if events:
                formatted = engine.expression.format_event(events[0])
                response.update(formatted)
                response["camera_online"] = is_cam_online
        except Exception as e:
            print(f"⚠️ 读取数据库出错: {e}")

    return response


@app.get("/events")
def get_events(limit: int = 50):
    if not engine or not engine.db:
        return []
    return engine.db.get_recent_events(limit=limit)


@app.get("/rhythm")
def get_rhythm(limit: int = 96):
    if not engine or not engine.db:
        return []
    return engine.db.get_recent_rhythm(limit=limit)


@app.get("/metrics/latest")
def get_latest_metrics(limit: int = 100):
    if not engine or not engine.db:
        return []
    return engine.db.get_recent_metrics(limit=limit)


@app.get("/metrics/summary")
def get_metrics_summary(limit: int = 100):
    if not engine or not engine.db:
        return {}
    return engine.db.get_metrics_summary(limit=limit)


@app.post("/feedback")
def feedback(payload: FeedbackPayload):
    if not engine or not engine.db:
        return {"ok": False, "error": "engine not ready"}
    feedback_id = engine.db.save_hard_sample(payload.event_id, payload.feedback_type, payload.note)
    return {"ok": True, "feedback_id": feedback_id}


@app.get("/llm/config")
def get_llm_config():
    return public_config()


@app.post("/llm/config")
def set_llm_config(payload: LLMConfigPayload):
    config = payload.model_dump()
    if not config.get("api_key"):
        current = public_config()
        config.pop("api_key", None)
    return {"ok": True, "config": save_config(config)}


@app.post("/generate_feedback")
def generate_feedback(payload: GenerateFeedbackPayload):
    if not engine or not engine.db:
        return {"ok": False, "error": "engine not ready"}
    event = payload.event
    if event is None and payload.event_id is not None:
        events = engine.db.get_recent_events(limit=200)
        event = next((e for e in events if e.get("id") == payload.event_id), None)
    if event is None:
        events = engine.db.get_recent_events(limit=1)
        event = events[0] if events else None
    if not event:
        return {"ok": False, "error": "no event available"}
    result = generate_event_feedback(event)
    result["event"] = event
    return result


@app.post("/chat")
def chat(payload: ChatPayload):
    if not engine or not engine.db:
        return {"ok": False, "error": "engine not ready"}
    context = engine.db.get_context_events(limit=payload.limit)
    result = answer_health_question(payload.messages, context)
    result["context"] = context
    return result


@app.get("/community/high_risk")
def community_high_risk(limit: int = 50):
    if not engine or not engine.db:
        return []
    return engine.db.get_high_risk_events(limit=limit)


@app.post("/community/action")
def community_action(payload: CommunityActionPayload):
    if not engine or not engine.db:
        return {"ok": False, "error": "engine not ready"}
    feedback_id = engine.db.save_community_feedback(payload.event_id, payload.action, payload.operator)
    return {"ok": True, "feedback_id": feedback_id}


@app.post("/demo/video")
async def demo_video(file: UploadFile = File(...), force_vlm: bool = False, persist: bool = True):
    if not engine:
        return {"ok": False, "error": "engine not ready"}

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        frames = load_video_frames(tmp_path, num_frames=8)
        if len(frames) < 1:
            return {"ok": False, "error": "no readable frames", "filename": file.filename}
        while len(frames) < 8:
            frames.append(frames[-1].copy())
        result = engine.process_frames(frames[:8], persist=persist, force_vlm=force_vlm)
        result["filename"] = file.filename
        result["frames_used"] = len(frames[:8])
        result["persisted"] = persist
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    print(f"🚀 服务启动中... (Root: {project_root})")
    uvicorn.run(app, host="0.0.0.0", port=8000)
