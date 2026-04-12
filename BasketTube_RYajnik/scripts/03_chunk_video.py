import cv2
from pathlib import Path
from tqdm import tqdm
import subprocess
import json

# ========================= CONFIG =========================
VIDEO_PATH = "data/raw/basketball_game.mp4"
OUTPUT_DIR = "outputs/clips"
CHUNK_DURATION = 20          # seconds per chunk
OVERLAP = 3                  # seconds overlap between chunks
# =======================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"📂 Processing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"✅ Video loaded: {duration/60:.1f} minutes @ {fps:.2f} FPS")

# Fixed-length chunking
chunk_id = 0
start_time = 0.0

print("🔪 Starting video chunking...")

while start_time < duration:
    end_time = min(start_time + CHUNK_DURATION, duration)
    
    output_path = f"{OUTPUT_DIR}/chunk_{chunk_id:04d}_{start_time:.1f}-{end_time:.1f}s.mp4"
    
    # Use ffmpeg for fast and high-quality splitting
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_time), "-i", VIDEO_PATH,
        "-t", str(CHUNK_DURATION + OVERLAP),
        "-c", "copy", "-avoid_negative_ts", "make_zero",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"✅ Created chunk {chunk_id:04d}: {start_time:.1f}s - {end_time:.1f}s")
    
    start_time += CHUNK_DURATION - OVERLAP
    chunk_id += 1

# Save metadata
metadata = {
    "video_duration": round(duration, 2),
    "total_chunks": chunk_id,
    "chunk_duration": CHUNK_DURATION,
    "overlap": OVERLAP,
    "chunks": [f"chunk_{i:04d}" for i in range(chunk_id)]
}

with open("outputs/results/video_chunks.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n🎉 Chunking completed! {chunk_id} clips created in outputs/clips/")
print(f"Metadata saved to outputs/results/video_chunks.json")
