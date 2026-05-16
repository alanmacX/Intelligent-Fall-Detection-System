import cv2
import time
import threading
import logging


class PerceptionModule:
    """
    Video stream reader and hardware status monitor.
    """

    def __init__(self, source="rtsp://192.168.31.120:8554/stream", resize_width=224, loop_file=False):
        self.source = source
        self.resize_width = resize_width
        self.loop_file = loop_file

        self.cap = None
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()
        self.reconnect_interval = 5

        self._connect()

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        logging.info(f"🎥 [感知层] 正在连接视频源: {self.source}")
        if self.cap:
            self.cap.release()

        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logging.error(f"❌ [感知层] 无法打开视频源: {self.source}")
                self.grabbed = False
            else:
                logging.info("✅ [感知层] 视频源连接成功")
                self.grabbed = True
        except Exception as e:
            logging.error(f"❌ [感知层] 连接异常: {e}")
            self.grabbed = False

    def _update_loop(self):
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                self.grabbed = False
                time.sleep(self.reconnect_interval)
                self._connect()
                continue

            ret, frame = self.cap.read()

            if not ret:
                self.grabbed = False
                if self.loop_file and isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning("⚠️ [感知层] 视频流中断，尝试重连...")
                    self.cap.release()
                    time.sleep(self.reconnect_interval)
                    self._connect()
                    continue

            if self.resize_width:
                h, w = frame.shape[:2]
                scale = self.resize_width / w
                frame = cv2.resize(frame, (self.resize_width, int(h * scale)))

            with self.lock:
                self.frame = frame
                self.grabbed = True

            time.sleep(0.03)

    def read(self):
        """Return the latest frame."""
        with self.lock:
            return self.frame.copy() if self.grabbed and self.frame is not None else None

    def is_online(self):
        """Return whether the video source is currently readable."""
        return self.grabbed and self.cap is not None and self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join()
        if self.cap: self.cap.release()
