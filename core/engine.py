import time
import threading
import logging
import numpy as np
from core.perception import PerceptionModule
from core.storage import StorageEngine
from core.expression import ExpressionEngine
from core.cognition_backup import GuardianCognition, CLASS_LABELS

# === 逻辑树参数 ===
ADL_BENDING_SCALE = 0.65
ADL_LYING_SCALE = 0.9
FALL_SLOW_SCALE = 1.8
FALL_IDXS = [6, 7, 8, 9, 10, 11]  # 对应 CLASS_LABELS 的后6个
ADL_IDXS = [0, 1, 2, 3, 4, 5]

# 理由汉化表
REASON_MAP = {
    "CLIP Decision": "端侧模型判定",
    "Low Confidence": "置信度过低",
    "Pattern Match": "高危行为模式匹配",
    "FastVLM Confirmed": "云端大模型二次确认"
}


class GuardianEngine:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("🚀 [引擎] 系统启动...")

        # ⚠️ 注意：这里 source=0 是默认摄像头，若要用手机流请改这里
        self.perception = PerceptionModule(source="rtsp://admin:123456@192.168.31.120:8554/live")
        self.db = StorageEngine()
        self.expression = ExpressionEngine()
        self.cognition = GuardianCognition()

        self.running = True
        self.frame_buffer = []

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            frame = self.perception.read()
            if frame is None:
                time.sleep(0.05)
                continue  # 摄像头没数据，循环等待

            # 维护 8 帧
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > 8: self.frame_buffer.pop(0)
            if len(self.frame_buffer) < 8: continue

            # A. 推理
            probs = self.cognition.infer_actionclip(self.frame_buffer)
            if probs is None: continue

            # B. 加权 (简化版逻辑树)
            weights = np.ones_like(probs)
            for i, label in enumerate(CLASS_LABELS):
                l = label.lower()
                if "bending" in l: weights[i] *= ADL_BENDING_SCALE
                if "lying" in l: weights[i] *= ADL_LYING_SCALE
                if "slow" in l: weights[i] *= FALL_SLOW_SCALE

            weighted_probs = probs * weights
            fall_score = np.sum(weighted_probs[FALL_IDXS])
            adl_score = np.sum(weighted_probs[ADL_IDXS])

            # C. 判决
            decision = "ADL"
            reason_en = "CLIP Decision"
            vlm_desc = ""
            is_active = False

            if fall_score > adl_score:
                decision = "FALL"
                # 如果是跌倒，调用 VLM 描述
                if self.cognition.vlm_model:
                    is_active = True
                    _, vlm_desc = self.cognition.infer_fastvlm(frame)
                    reason_en = "FastVLM Confirmed"
            elif abs(fall_score - adl_score) < 0.15:
                # 模糊地带
                decision = "WARNING"
                reason_en = "Pattern Match"

            # D. 存库
            cn_reason = REASON_MAP.get(reason_en, reason_en)
            display_label = "FALL" if decision == "FALL" else "SAFE"
            if decision == "WARNING": display_label = "UNKNOWN"

            if decision == "FALL" and not vlm_desc:
                vlm_desc = f"端侧检测到跌倒。判定依据：{cn_reason}"

            self.db.save_event(display_label, max(fall_score, adl_score), vlm_desc, is_active)

            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.perception.release()