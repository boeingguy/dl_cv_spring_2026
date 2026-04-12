from pathlib import Path
import json
from ollama import chat
import base64
from tqdm import tqdm

# ========================= CONFIG =========================
DETECTIONS_PATH = "outputs/results/detections/player_detections.json"
RESULTS_PATH = "outputs/results/vlm_action_analysis.json"
LLM_MODEL = "qwen2.5:14b"      
MAX_SAMPLES = 30               
# =======================================================

with open(DETECTIONS_PATH, "r") as f:
    detections = json.load(f)

print(f"Loaded {len(detections):,} detections")

# Select diverse samples
samples = detections[::max(1, len(detections)//MAX_SAMPLES)][:MAX_SAMPLES]

results = []

for det in tqdm(samples, desc="Analyzing with VLM"):
    frame_path = Path("outputs") / det["frame_path"]
    
    if not frame_path.exists():
        continue

    # Read and encode image
    with open(frame_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = """You are an expert basketball analyst. Look at the image and describe:

1. What is the main player doing? (shooting, passing, dribbling, defending, rebounding, cutting, standing, etc.)
2. Body posture and action clarity
3. Approximate court location (near basket, three-point line, mid-court, paint, etc.)
4. Any visible jersey number or team if clear

Be concise and direct."""

    response = chat(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_base64]          # This is the correct format
        }]
    )

    analysis = response['message']['content']

    results.append({
        "chunk": det["chunk"],
        "frame": det["frame"],
        "bbox": det["bbox"],
        "vlm_description": analysis.strip(),
        "timestamp_estimate": round(det["frame"] / 30, 2)
    })

# Save results
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ VLM Analysis completed! {len(results)} frames analyzed.")
print(f"Results saved to: {RESULTS_PATH}")
