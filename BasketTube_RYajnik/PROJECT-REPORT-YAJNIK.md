# BasketTube_RYajnik - Basketball Game Analysis

**Final Project** — Deep Learning in Computer Vision (Spring 2026)  
**Author**: Ruchik Yajnik

---

## Project Overview

This project implements a **dual-approach** system for analyzing a basketball game:

- **Task 1**: Analyze player performance using **spoken commentary** only.
- **Task 2**: Verify actual player actions by analyzing the **video footage** directly.

The goal is to compare the strengths and limitations of audio-based vs. vision-based sports analytics.

---

## Task 1 – Commentary-Based Analysis

### Approach & Tools
- **Transcription**: Faster-Whisper (`large-v3`) with word-level timestamps
- **Retrieval**: Sentence-Transformers (`all-MiniLM-L6-v2`) + cosine similarity
- **LLM**: Qwen2.5-14B (local via Ollama)
- Strict prompting to enforce structured, timestamped, grounded answers

### Key Features
- Google-style responses
- Timestamp citations
- Clear separation of facts vs. interpretations
- Required disclaimer on every answer

**Main Notebook**: [`Task1_Commentary_Analysis.ipynb`](notebooks/Task1_Commentary_Analysis.ipynb)

---

## Task 2 – Video-Based Action Verification

### Task 2.1: Video Chunking
- Fixed-length sliding windows (**20 seconds** with **3s overlap**)
- Re-encoded all clips from AV1 → H.264 using FFmpeg
- **Total chunks created**: **332**

### Task 2.2: Player Detection & Action Analysis
- **Player Detection**: YOLOv8x on keyframes (~every 1.5 seconds) → **11,848 detections**
- **Action Recognition**: Qwen2.5-14B Vision-Language Model
- Focused actions: shooting, dribbling, passing, rebounding, defending

**Main Notebook**: [`task2.ipynb`](notebooks/task2.ipynb)

**Representative frames** are available in `outputs/frames/`

---

## Comparison: Task 1 vs Task 2

| Aspect                    | Task 1 (Commentary)                        | Task 2 (Video)                                | Observation |
|--------------------------|-------------------------------------------|----------------------------------------------|-----------|
| Speed                    | Very fast                                 | Computationally intensive                    | Task 1 wins |
| Context & Narrative      | Excellent                                 | Limited                                      | Task 1 wins |
| Objectivity              | Low (bias-prone)                          | High (visual evidence)                       | Task 2 wins |
| Specific Actions         | Good high-level                           | Better at concrete actions                   | Task 2 wins |
| Fine Details             | Strong                                    | Weak (occlusion, small ball)                 | Task 1 wins |
| Evidence Type            | Verbal + timestamps                       | Visual frames + timestamps                   | Complementary |

**Key Insights**:
- **Agreement**: Both methods correctly identified major events (e.g., LeBron’s eye poke, shooting attempts).
- **Divergence**: Commentary often exaggerated performance, while video showed more realistic actions.
- **Best Future Approach**: **Hybrid system** combining commentary context with visual verification.

---

## Limitations

**Task 1**:
- Dependent on commentator accuracy and bias
- Cannot verify visual reality

**Task 2**:
- Small/fast ball and heavy occlusion
- Camera cuts and changing angles
- No jersey number recognition
- Limited temporal reasoning for complex plays

---

## Conclusion & Future Work

This project successfully demonstrated the strengths and weaknesses of two fundamentally different approaches to sports analytics.

While commentary analysis is excellent for narrative and quick insights, video analysis provides more objective grounding — though it is currently limited by technical challenges in computer vision.

**With more time, I would**:
- Implement robust player tracking (ByteTrack + ReID)
- Add ball detection and trajectory analysis
- Use jersey number OCR
- Build a true hybrid commentary + vision system
- Explore stronger video-language models

---

**Repository**: [`BasketTube_RYajnik`](.)

**Main Deliverables**:
- `notebooks/Task1_Commentary_Analysis.ipynb`
- `notebooks/task2.ipynb`
- Full processing scripts in `/scripts/`

---

Thank you for reviewing!
