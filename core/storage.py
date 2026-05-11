import sqlite3
import time
import os
from threading import Lock


class StorageEngine:
    def __init__(self, db_path="core/data/guardian.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # 允许通过列名访问
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
            self.conn.commit()

    def save_event(self, raw_label, confidence, vlm_desc, is_router_active):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.cursor.execute('''
                INSERT INTO events (timestamp, raw_label, confidence, vlm_description, is_router_active)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, raw_label, confidence, vlm_desc, 1 if is_router_active else 0))
            self.conn.commit()

    def get_recent_events(self, limit=1):
        with self.lock:
            self.cursor.execute('SELECT * FROM events ORDER BY id DESC LIMIT ?', (limit,))
            return [dict(row) for row in self.cursor.fetchall()]

    def close(self):
        self.conn.close()