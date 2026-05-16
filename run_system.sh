#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🚀 启动后端 API..."
nohup conda run -n fall python api/server.py > api.log 2>&1 &
API_PID=$!

echo "⏳ 等待模型加载 (15s)..."
sleep 15

trap "kill $API_PID" EXIT

echo "🚀 启动原生 Web 前端..."
cd web_demo || exit 1
python -m http.server 5173
