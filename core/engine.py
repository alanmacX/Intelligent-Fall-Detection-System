import time
import threading
import logging
import os
import numpy as np
import torch
from core.perception import PerceptionModule
from core.storage import StorageEngine
from core.expression import ExpressionEngine
from core.cognition_backup import GuardianCognition, CLASS_LABELS
from core.rhythm import RhythmRuntime

ADL_BENDING_SCALE = 0.65
ADL_LYING_SCALE = 0.9
FALL_SLOW_SCALE = 1.8
FALL_IDXS = [6, 7, 8, 9, 10, 11]
ADL_IDXS = [0, 1, 2, 3, 4, 5]

REASON_MAP = {
    "CLIP Decision": "端侧模型判定",
    "Low Confidence": "置信度过低",
    "Pattern Match": "高危行为模式匹配",
    "FastVLM Confirmed": "云端大模型二次确认"
}


def now_ms():
    return time.perf_counter() * 1000.0


def route_policy(router_score, probs, stats, rhythm):
    fall_mass = float(np.sum(probs[FALL_IDXS]))
    reasons = []
    has_visual_risk = fall_mass >= 0.18 or stats["entropy"] >= 1.2 or stats["margin"] <= 0.25
    if router_score >= 0.8 and has_visual_risk:
        reasons.append("router_high")
    if router_score >= 0.5 and fall_mass >= 0.35:
        reasons.append("fall_mass")
    if router_score >= 0.5 and stats["entropy"] >= 1.2:
        reasons.append("high_entropy")
    if router_score >= 0.5 and stats["margin"] <= 0.25:
        reasons.append("low_margin")
    if router_score >= 0.5 and rhythm["surprise"] >= 1.35 and fall_mass >= 0.18:
        reasons.append("rhythm_fall_context")
    return bool(reasons), reasons, fall_mass


class GuardianEngine:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("🚀 [引擎] 系统启动...")

        source = os.environ.get("GUARDIAN_VIDEO_SOURCE", "rtsp://admin:123456@192.168.31.120:8554/live")
        self.perception = PerceptionModule(source=source)
        self.db = StorageEngine()
        self.expression = ExpressionEngine()
        self.cognition = GuardianCognition()
        self.rhythm = RhythmRuntime()

        self.running = True
        self.frame_buffer = []

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            frame = self.perception.read()
            if frame is None:
                time.sleep(0.05)
                continue

            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > 8: self.frame_buffer.pop(0)
            if len(self.frame_buffer) < 8: continue

            self.process_frames(self.frame_buffer, persist=True)

            time.sleep(0.1)

    def process_frames(self, frames, persist=True, force_vlm=False):
        total_start = now_ms()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = now_ms()
        result = self.cognition.infer_actionclip(frames, return_features=True)
        actionclip_ms = now_ms() - t0
        if result is None:
            return {"ok": False, "error": "ActionCLIP unavailable"}
        probs, feats, stats = result

        top_idx = int(np.argmax(probs))
        actionclip_label = CLASS_LABELS[top_idx]
        actionclip_conf = float(probs[top_idx])

        t0 = now_ms()
        rhythm = self.rhythm.observe(probs)
        rhythm_ms = now_ms() - t0

        t0 = now_ms()
        router_score, router_uncertainty, router_raw = self.cognition.bayesian_route(
            feats, rhythm["surprise"], samples=5
        )
        router_ms = now_ms() - t0

        should_route, route_reasons, fall_mass = route_policy(router_score, probs, stats, rhythm)
        vlm_used = False
        vlm_desc = ""
        vlm_label = None
        vlm_ms = 0.0
        final_label = "FALL" if top_idx in FALL_IDXS else "SAFE"
        source = "ActionCLIP + RhythmRouter"

        if (force_vlm or should_route) and self.cognition.vlm_model:
            t0 = now_ms()
            vlm_label, vlm_desc = self.cognition.infer_fastvlm(frames)
            vlm_ms = now_ms() - t0
            vlm_used = True
            if vlm_label in ("FALL", "SAFE"):
                final_label = vlm_label
            source = "FastVLM Confirmed" if should_route else "FastVLM Forced"

        if not vlm_desc:
            vlm_desc = (
                f"{source}: {actionclip_label}, conf={actionclip_conf:.3f}, "
                f"router={router_score:.3f}, rhythm_surprise={rhythm['surprise']:.3f}"
            )

        t0 = now_ms()
        event_id = None
        if persist:
            event_id = self.db.save_event(
                final_label,
                actionclip_conf,
                vlm_desc,
                vlm_used,
                router_score=router_score,
                router_uncertainty=router_uncertainty,
                rhythm_surprise=rhythm["surprise"],
                entropy=stats["entropy"],
                margin=stats["margin"],
                privacy_mask=True,
            )
            self.db.save_rhythm(
                CLASS_LABELS[rhythm["top_idx"]],
                rhythm["hour"],
                rhythm["activity_score"],
                rhythm["surprise"],
                rhythm["anomaly"],
            )
        storage_ms = now_ms() - t0

        allocated = reserved = peak = 0.0
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        metrics = {
            "actionclip_ms": actionclip_ms,
            "rhythm_ms": rhythm_ms,
            "router_ms": router_ms,
            "vlm_ms": vlm_ms,
            "storage_ms": storage_ms,
            "total_ms": now_ms() - total_start,
            "vlm_used": vlm_used,
            "gpu_mem_allocated_mb": allocated,
            "gpu_mem_reserved_mb": reserved,
            "gpu_mem_peak_mb": peak,
        }
        if persist:
            self.db.save_metrics(metrics)

        top_indices = np.argsort(probs)[::-1][:5]
        return {
            "ok": True,
            "event_id": event_id,
            "final_label": final_label,
            "source": source,
            "vlm_used": vlm_used,
            "vlm_label": vlm_label,
            "vlm_text": vlm_desc,
            "actionclip": {
                "top_label": actionclip_label,
                "top_index": top_idx,
                "confidence": actionclip_conf,
                "entropy": stats["entropy"],
                "margin": stats["margin"],
                "probabilities": [
                    {"index": i, "label": CLASS_LABELS[i], "probability": float(probs[i])}
                    for i in range(len(CLASS_LABELS))
                ],
                "top5": [
                    {"index": int(i), "label": CLASS_LABELS[int(i)], "probability": float(probs[int(i)])}
                    for i in top_indices
                ],
            },
            "rhythm": {
                "model": rhythm.get("model"),
                "hour": rhythm["hour"],
                "surprise": rhythm["surprise"],
                "kl_raw": rhythm.get("kl_raw"),
                "anomaly": rhythm["anomaly"],
                "prior_top": CLASS_LABELS[int(np.argmax(rhythm["prior"]))],
                "prior_probabilities": [
                    {"index": i, "label": CLASS_LABELS[i], "probability": float(rhythm["prior"][i])}
                    for i in range(len(CLASS_LABELS))
                ],
            },
            "router": {
                "score": router_score,
                "raw_score": router_raw,
                "uncertainty": router_uncertainty,
                "threshold": 0.5,
                "should_route": should_route,
                "route_reasons": route_reasons,
                "fall_mass": fall_mass,
                "force_vlm": force_vlm,
            },
            "metrics": metrics,
        }

    def stop(self):
        self.running = False
        self.perception.release()
