import sqlite3
import time
import os
from threading import Lock
import numpy as np


class StorageEngine:
    def __init__(self, db_path="core/data/guardian.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.lock = Lock()
        self._init_tables()

    def _init_tables(self):
        with self.lock:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    raw_label TEXT,
                    confidence REAL,
                    vlm_description TEXT,
                    is_router_active INTEGER
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS rhythm_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    hour INTEGER,
                    label TEXT,
                    activity_score REAL,
                    surprise REAL,
                    anomaly REAL
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS hard_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_id INTEGER,
                    feedback_type TEXT,
                    note TEXT
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS inference_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    actionclip_ms REAL,
                    rhythm_ms REAL,
                    router_ms REAL,
                    vlm_ms REAL,
                    storage_ms REAL,
                    total_ms REAL,
                    vlm_used INTEGER,
                    gpu_mem_allocated_mb REAL,
                    gpu_mem_reserved_mb REAL,
                    gpu_mem_peak_mb REAL
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS community_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER,
                    action TEXT,
                    operator TEXT,
                    timestamp TEXT
                )
            ''')
            self._ensure_event_columns()
            self.conn.commit()

    def _ensure_event_columns(self):
        self.cursor.execute("PRAGMA table_info(events)")
        columns = {row[1] for row in self.cursor.fetchall()}
        additions = {
            "router_score": "REAL",
            "router_uncertainty": "REAL",
            "rhythm_surprise": "REAL",
            "entropy": "REAL",
            "margin": "REAL",
            "privacy_mask": "INTEGER DEFAULT 1",
        }
        for name, column_type in additions.items():
            if name not in columns:
                self.cursor.execute(f"ALTER TABLE events ADD COLUMN {name} {column_type}")

    def save_event(
        self,
        raw_label,
        confidence,
        vlm_desc,
        is_router_active,
        router_score=None,
        router_uncertainty=None,
        rhythm_surprise=None,
        entropy=None,
        margin=None,
        privacy_mask=True,
    ):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.cursor.execute('''
                INSERT INTO events (
                    timestamp, raw_label, confidence, vlm_description, is_router_active,
                    router_score, router_uncertainty, rhythm_surprise, entropy, margin, privacy_mask
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, raw_label, confidence, vlm_desc, 1 if is_router_active else 0,
                router_score, router_uncertainty, rhythm_surprise, entropy, margin, 1 if privacy_mask else 0
            ))
            self.conn.commit()
            return self.cursor.lastrowid

    def save_rhythm(self, label, hour, activity_score, surprise, anomaly):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.cursor.execute('''
                INSERT INTO rhythm_events (timestamp, hour, label, activity_score, surprise, anomaly)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, hour, label, activity_score, surprise, anomaly))
            self.conn.commit()

    def get_recent_events(self, limit=1):
        with self.lock:
            self.cursor.execute('SELECT * FROM events ORDER BY id DESC LIMIT ?', (limit,))
            return [self._clean_row(dict(row)) for row in self.cursor.fetchall()]

    def _clean_row(self, row):
        clean = {}
        for key, value in row.items():
            if isinstance(value, bytes):
                if len(value) == 4:
                    clean[key] = float(np.frombuffer(value, dtype=np.float32)[0])
                elif len(value) == 8:
                    clean[key] = float(np.frombuffer(value, dtype=np.float64)[0])
                else:
                    clean[key] = value.hex()
            elif isinstance(value, np.generic):
                clean[key] = value.item()
            else:
                clean[key] = value
        return clean

    def get_recent_rhythm(self, limit=96):
        with self.lock:
            self.cursor.execute('SELECT * FROM rhythm_events ORDER BY id DESC LIMIT ?', (limit,))
            rows = [self._clean_row(dict(row)) for row in self.cursor.fetchall()]
            return list(reversed(rows))

    def save_hard_sample(self, event_id, feedback_type, note=""):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.cursor.execute('''
                INSERT INTO hard_samples (timestamp, event_id, feedback_type, note)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, event_id, feedback_type, note))
            self.conn.commit()
            return self.cursor.lastrowid

    def save_community_feedback(self, event_id, action, operator="demo_operator"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.cursor.execute('''
                INSERT INTO community_feedback (event_id, action, operator, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (event_id, action, operator, timestamp))
            if action == "标记误报":
                self.cursor.execute('''
                    INSERT INTO hard_samples (timestamp, event_id, feedback_type, note)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, event_id, "false_positive", "community feedback"))
            self.conn.commit()
            return self.cursor.lastrowid

    def get_high_risk_events(self, limit=50):
        with self.lock:
            self.cursor.execute('''
                SELECT * FROM events
                WHERE raw_label = 'FALL'
                   OR is_router_active = 1
                   OR COALESCE(rhythm_surprise, 0) >= 1.2
                ORDER BY id DESC LIMIT ?
            ''', (limit,))
            return [self._clean_row(dict(row)) for row in self.cursor.fetchall()]

    def get_context_events(self, limit=30):
        events = list(reversed(self.get_recent_events(limit=limit)))
        lines = []
        for event in events:
            lines.append(
                f"{event.get('timestamp')} | label={event.get('raw_label')} | "
                f"conf={event.get('confidence')} | router={event.get('router_score')} | "
                f"rhythm={event.get('rhythm_surprise')} | vlm={event.get('is_router_active')} | "
                f"desc={event.get('vlm_description') or ''}"
            )
        return "\n".join(lines) if lines else "暂无历史事件。"

    def save_metrics(self, metrics):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.cursor.execute('''
                INSERT INTO inference_metrics (
                    timestamp, actionclip_ms, rhythm_ms, router_ms, vlm_ms, storage_ms, total_ms,
                    vlm_used, gpu_mem_allocated_mb, gpu_mem_reserved_mb, gpu_mem_peak_mb
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                metrics.get("actionclip_ms"),
                metrics.get("rhythm_ms"),
                metrics.get("router_ms"),
                metrics.get("vlm_ms"),
                metrics.get("storage_ms"),
                metrics.get("total_ms"),
                1 if metrics.get("vlm_used") else 0,
                metrics.get("gpu_mem_allocated_mb"),
                metrics.get("gpu_mem_reserved_mb"),
                metrics.get("gpu_mem_peak_mb"),
            ))
            self.conn.commit()

    def get_recent_metrics(self, limit=100):
        with self.lock:
            self.cursor.execute('SELECT * FROM inference_metrics ORDER BY id DESC LIMIT ?', (limit,))
            rows = [self._clean_row(dict(row)) for row in self.cursor.fetchall()]
            return list(reversed(rows))

    def get_metrics_summary(self, limit=100):
        rows = self.get_recent_metrics(limit=limit)
        if not rows:
            return {}
        total = [r["total_ms"] for r in rows if r.get("total_ms") is not None]
        vlm_used = [r["vlm_used"] for r in rows]
        total_sorted = sorted(total)
        p95_idx = min(len(total_sorted) - 1, int(0.95 * (len(total_sorted) - 1))) if total_sorted else 0
        return {
            "count": len(rows),
            "avg_total_ms": sum(total) / len(total) if total else 0.0,
            "p95_total_ms": total_sorted[p95_idx] if total_sorted else 0.0,
            "vlm_rate": sum(vlm_used) / len(vlm_used) if vlm_used else 0.0,
            "latest": rows[-1],
        }

    def close(self):
        self.conn.close()
