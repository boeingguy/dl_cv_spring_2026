# BasketTube_RYajnik

**Final Project** — Deep Learning in Computer Vision (Spring 2026)

A dual-approach basketball game analysis system that contrasts **commentary analysis** with **video-based action verification**.

---

## Project Overview

- **Task 1**: Chat-based assistant that analyzes player performance **using only game commentary**.
- **Task 2**: Verify actual player actions by analyzing the **video** using Computer Vision and Vision-Language Models.

---

## Project Structure

BasketTube_RYajnik/
├── data/
│   ├── raw/                    # Original basketball game video
│   └── processed/              # Transcript
├── outputs/
│   ├── clips/                  # Video chunks
│   ├── clips_reencoded/        # H.264 clips
│   ├── frames/                 # Extracted keyframes
│   └── results/                # Detections & VLM results
├── notebooks/
│   ├── Task1_Commentary_Analysis.ipynb
│   └── Task2_Video_Action_Analysis.ipynb
├── scripts/                    # Processing scripts
├── .venv/                      # Virtual environment
└── README.md



---

## Delivered Notebooks

| Notebook | Task | Description |
|---------|------|-----------|
| `Task1_Commentary_Analysis.ipynb` | Task 1 | Full commentary RAG system with structured Google-style answers |
| `Task2_Video_Action_Analysis.ipynb` | Task 2 | Video chunking + YOLOv8x detection + Qwen2.5 VLM action analysis |

---

## 🛠️ech Stack

- **LLM / VLM**: Qwen2.5-14B (via Ollama)
- **Transcription**: Faster-Whisper (`large-v3`)
- **Detection**: YOLOv8x
- **Retrieval**: Sentence-Transformers
- **Hardware**: RTX 5070 Ti

## Link to YouTube Video Demo
https://www.youtube.com/watch?v=n0EfZt1pyvs

---

## How to Run

1. Activate the environment:
   ```bash
   cd ~/BasketTube_RYajnik
   source .venv/bin/activate


2. Start Ollama (in another terminal):

	ollama serve

3. Open Jupyter Lab:

	jupyter lab

Results Summary

Task 1: Robust RAG system with timestamped evidence and disclaimers.
Task 2: 11,848 player detections + VLM action interpretation on 40+ frames.
Strong comparison between audio-based and vision-based analysis.
