from pathlib import Path
import subprocess
from tqdm import tqdm

CHUNKS_DIR = Path("outputs/clips")
REENCODED_DIR = Path("outputs/clips_reencoded")

REENCODED_DIR.mkdir(parents=True, exist_ok=True)

clips = sorted(CHUNKS_DIR.glob("*.mp4"))

print(f"Re-encoding {len(clips)} clips to H.264...")

for clip in tqdm(clips):
    output_path = REENCODED_DIR / clip.name
    
    if output_path.exists():
        continue
        
    subprocess.run([
        "ffmpeg", "-y", "-i", str(clip),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac",
        str(output_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ Re-encoding completed!")
print(f"New clips are in: {REENCODED_DIR}")
