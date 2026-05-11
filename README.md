# Intelligent Fall Detection System

An intelligent fall detection and monitoring system based on a two-stage video understanding pipeline. The system combines a lightweight ActionCLIP-based perception module, a Bayesian router, and a FastVLM-based visual-language verification module to detect high-risk fall events while reducing unnecessary large-model inference.

The project also provides a FastAPI backend and a Streamlit dashboard for real-time monitoring, event display, and historical record inspection.

## Overview

This system is designed for fall detection in indoor monitoring scenarios. It first performs fast action recognition on sampled video frames. When the lightweight model produces uncertain or high-risk results, a router decides whether to activate a stronger vision-language model for secondary verification.

The core idea is:

1. Use ActionCLIP for fast video-level action classification.
2. Use a lightweight router to decide whether the case needs further verification.
3. Use FastVLM to perform semantic reasoning on key frames when necessary.
4. Store and display monitoring events through a backend API and dashboard.

This makes the system more efficient than always calling a large vision-language model, while still preserving higher reliability for ambiguous or dangerous cases.

## Main Features

- Video-based fall detection
- 12-class action prompt design, including normal daily activities and multiple fall-related states
- ActionCLIP-based fast perception
- Bayesian lightweight router for adaptive VLM activation
- FastVLM-based secondary verification
- FastAPI backend service
- Streamlit monitoring dashboard
- SQLite-based event history display
- Video frame visualization utility for reports or patent-style figures

## Project Structure

```text
Intelligent-Fall-Detection-System/
├── api/
│   └── server.py              # FastAPI backend service
├── configs/
│   └── custom.yaml            # ActionCLIP/custom dataset configuration
├── core/
│   └── ...                    # Core engine, perception, database, and system modules
├── frontend/
│   ├── dashboard.py           # Streamlit dashboard
│   ├── components.py          # UI components
│   └── styles.css             # Dashboard styling
├── lib/
│   └── ...                    # External libraries or model dependencies
├── demo.py                    # Standalone inference demo
├── train_router.py            # Lightweight router training script
├── video_visualization.py     # Video frame visualization tool
├── run_system.sh              # One-click backend + dashboard launcher
└── README.md
