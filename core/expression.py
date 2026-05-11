import time


class ExpressionEngine:
    """
    表达引擎：中英文映射与 iOS 格式化
    """

    def __init__(self):
        self.STATUS_MAPPING = {
            "FALL": "⚠️ 跌倒报警",
            "SAFE": "✅ 状态安全",
            "BENDING": "弯腰 (低风险)",
            "LYING": "躺卧 (需关注)",
            "SITTING": "静坐",
            "STANDING": "站立",
            "WALKING": "行走",
            "UNKNOWN": "❓ 状态未知"
        }

        self.RISK_LEVEL_MAPPING = {
            "FALL": "high", "LYING": "medium", "BENDING": "low",
            "SAFE": "low", "SITTING": "low", "UNKNOWN": "low"
        }

    def format_event(self, raw_event):
        """将数据库原始记录转换为 iOS JSON"""
        if not raw_event:
            return {
                "title": "系统初始化", "risk_level": "low",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "System", "detail": "等待数据..."
            }

        label = raw_event.get("raw_label", "UNKNOWN")
        vlm_desc = raw_event.get("vlm_description", "")
        is_cloud = raw_event.get("is_router_active", False)

        return {
            "title": self.STATUS_MAPPING.get(label, label),
            "risk_level": self.RISK_LEVEL_MAPPING.get(label, "low"),
            "time": raw_event.get("timestamp", ""),
            "source": "【云端双重确认】" if is_cloud else "【端侧实时监测】",
            "detail": vlm_desc if vlm_desc else "行为特征正常，无需介入。",
            "suggestion": "请立即查看！" if label == "FALL" else "监测中..."
        }