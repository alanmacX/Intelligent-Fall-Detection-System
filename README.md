# Intelligent Fall Detection System

An intelligent fall detection and monitoring system based on a two-stage video understanding pipeline. The system combines ActionCLIP-based action recognition, RhythmMamba rhythm modeling, a Bayesian router, and FastVLM visual-language verification to detect high-risk fall events while reducing unnecessary large-model inference.

The project provides a FastAPI backend and a native HTML/CSS/JS Web demo for video inference, event monitoring, semantic feedback, rhythm analysis, and runtime performance inspection.

## Overview

The system is designed for indoor fall detection scenarios. It first performs fast action recognition on sampled video frames. When the lightweight model produces uncertain or high-risk results, the router decides whether to activate FastVLM for secondary semantic verification.

Core flow:

1. Use ActionCLIP for fast video-level action classification.
2. Use RhythmMamba to model activity rhythm and surprise.
3. Use a Bayesian router to decide whether the case needs VLM verification.
4. Use FastVLM to reason over key frames when necessary.
5. Store and display monitoring events, metrics, and semantic responses through the backend API and Web frontend.

## Main Features

- Video-based fall detection
- 12-class action prompt design for ADL and fall subclasses
- ActionCLIP-based fast perception
- RhythmMamba rhythm surprise modeling
- Bayesian router for adaptive VLM activation
- FastVLM-based secondary verification
- FastAPI backend service
- Native Web demo in `web_demo/`
- SQLite-based event and metric history
- Video frame visualization utility for reports

## Start

```bash
bash run_system.sh
```

Default services:

- Backend API: `http://127.0.0.1:8000`
- Native Web frontend: `http://127.0.0.1:5173`

The frontend entry is `web_demo/index.html`. The default launcher does not use Streamlit.

## Inference Metrics

The video demo displays:

- Final decision and route reasons
- ActionCLIP, RhythmMamba, Bayesian Router, FastVLM, and Storage stage latency
- End-to-end latency
- CUDA peak memory usage

## Dataset Config

The current training config uses the 12-class split:

```yaml
train_list: ../fall_dataset/list_12/train_list.txt
val_list: ../fall_dataset/list_12/val_list.txt
label_list: ../fall_dataset/list_12/label_list.csv
```

## Project Structure

```text
Intelligent-Fall-Detection-System/
├── api/                    # FastAPI backend and LLM client
├── configs/                # System and model configs
├── core/                   # Inference engine, perception, storage, rhythm, cognition
├── frontend/               # Legacy Streamlit dashboard
├── lib/                    # External model/library code
├── scripts/                # Dataset and rhythm/router data utilities
├── web_demo/               # Native HTML/CSS/JS frontend
├── demo.py                 # Standalone inference demo
├── train_router.py         # Router training script
├── train_router_rhythm.py  # Rhythm-aware router training
├── train_rhythm_mamba.py   # RhythmMamba training
├── video_visualization.py  # Visualization utility
├── run_system.sh           # One-click backend + Web frontend launcher
└── README.md
