import sys
import os

# ==========================================
# 🔥 核心路径修复 (必须放在最前面)
# ==========================================
# 获取当前文件 (api/server.py) 的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取 api 目录
api_dir = os.path.dirname(current_file_path)
# 获取项目根目录 (Intelligent_Fall_Detection_System)
project_root = os.path.dirname(api_dir)

# 强行把根目录塞进 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import time

# 现在可以安全导入 core 了
try:
    from core.engine import GuardianEngine
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"🔍 调试: sys.path[0] = {sys.path[0]}")
    sys.exit(1)

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # 初始化引擎
    engine = GuardianEngine()
    yield
    # 关闭引擎
    if engine:
        engine.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "running", "service": "Guardian AI"}


@app.get("/latest_event")
def get_latest_event():
    """
    返回最新监控数据 + 摄像头硬件状态
    """
    # 1. 检查硬件在线状态
    is_cam_online = False
    if engine and engine.perception:
        is_cam_online = engine.perception.is_online()

    # 2. 构造基础响应
    response = {
        "title": "系统启动中...",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera_online": is_cam_online,  # 🔥 关键字段：告诉前端摄像头挂没挂
        "risk_level": "low",
        "source": "System",
        "detail": "等待数据流接入...",
        "suggestion": "正在初始化..."
    }

    # 3. 尝试读取数据库最新记录
    if engine and engine.db:
        try:
            events = engine.db.get_recent_events(limit=1)
            if events:
                # 格式化数据 (使用 engine 里的 expression 模块)
                formatted = engine.expression.format_event(events[0])
                response.update(formatted)
                # 再次强制覆盖 camera_online，防止被 formatted 冲掉
                response["camera_online"] = is_cam_online
        except Exception as e:
            print(f"⚠️ 读取数据库出错: {e}")

    return response


if __name__ == "__main__":
    print(f"🚀 服务启动中... (Root: {project_root})")
    # host="0.0.0.0" 保证局域网可访问
    uvicorn.run(app, host="0.0.0.0", port=8000)