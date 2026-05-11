#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🚀 启动后端 API..."
nohup python api/server.py > api.log 2>&1 &
API_PID=$!

echo "⏳ 等待模型加载 (15s)..."
sleep 15

echo "🚀 启动前端 Dashboard..."
streamlit run frontend/dashboard.py

trap "kill $API_PID" EXIT