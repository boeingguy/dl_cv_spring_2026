import cv2
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm

# ========================= CONFIG =========================
CHUNKS_DIR = Path("outputs/clips_reencoded")
FRAMES_DIR = Path("outputs/frames")
DETECTIONS_DIR = Path("outputs/results/detections")

DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading YOLOv8x model...")
model = YOLO("yolov8x.pt")

# Process first 30 chunks for reasonable runtime (you can increase later)
chunk_files = sorted(CHUNKS_DIR.glob("*.mp4"))[:30]

print(f"Processing {len(chunk_files)} re-encoded chunks...\n")

all_detections = []

for chunk_path in tqdm(chunk_files, desc="Detecting players"):
    chunk_name = chunk_path.stem
    chunk_frame_dir = FRAMES_DIR / chunk_name
    chunk_frame_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(chunk_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    frame_count = 0
    detections_in_chunk = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample ~every 1.5 seconds
        if frame_count % max(1, int(fps * 1.5)) == 0:
            results = model(frame, conf=0.25, verbose=False)
            
            # Save frame
            frame_path = chunk_frame_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Save detections
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                all_detections.append({
                    "chunk": chunk_name,
                    "frame": frame_count,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 3),
                    "frame_path": str(frame_path.relative_to("outputs"))
                })
                detections_in_chunk += 1
        
        frame_count += 1
    
    cap.release()
    print(f"  → {chunk_name}: {detections_in_chunk} detections")

# Save all detections
with open(DETECTIONS_DIR / "player_detections.json", "w") as f:
    json.dump(all_detections, f, indent=2)

print(f"\n🎉 Player Detection Completed!")
print(f"   • Chunks processed : {len(chunk_files)}")
print(f"   • Total detections : {len(all_detections)}")
print(f"   • Frames saved to  : outputs/frames/")
