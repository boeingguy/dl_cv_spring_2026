from faster_whisper import WhisperModel
import json
from pathlib import Path
import subprocess
import sys
from tqdm import tqdm
import time

# ===================== CONFIG =====================
VIDEO_PATH = "data/raw/basketball_game.mp4"      # ← Change if your filename is different
AUDIO_PATH = "data/processed/audio.wav"
TRANSCRIPT_JSON = "data/processed/transcript.json"
TRANSCRIPT_TXT = "data/processed/transcript.txt"

MODEL_SIZE = "large-v3"      # Use "medium" for faster testing
# ================================================

def extract_audio(video_path: str, audio_path: str):
    print(f"🎥 Extracting audio from video...")
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", audio_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ Audio extracted successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to extract audio: {e}")
        return False

# Create directories
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Extract audio if it doesn't exist
if not Path(AUDIO_PATH).exists():
    if not Path(VIDEO_PATH).exists():
        print(f"❌ Video file not found: {VIDEO_PATH}")
        print("Please put your game video in data/raw/ and update the VIDEO_PATH")
        sys.exit(1)
    extract_audio(VIDEO_PATH, AUDIO_PATH)
else:
    print("✅ Using existing audio file")

# Load model
print(f"🤖 Loading Whisper model ({MODEL_SIZE}) on GPU...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

# Transcribe with progress
print("🔄 Starting transcription... (this may take 5-20 minutes)")

segments, info = model.transcribe(
    AUDIO_PATH,
    beam_size=5,
    word_timestamps=True,
    vad_filter=True,
    language="en"
)

print(f"🎤 Detected language: {info.language} (probability: {info.language_probability:.2f})")
print("📝 Transcribing...")

transcript = []
start_time = time.time()

# Add tqdm progress bar
for segment in tqdm(segments, desc="Transcribing segments", unit="seg"):
    entry = {
        "start": round(segment.start, 2),
        "end": round(segment.end, 2),
        "text": segment.text.strip()
    }
    transcript.append(entry)
    
    # Print live output
    print(f"[{entry['start']:>7.2f} → {entry['end']:>7.2f}s] {entry['text']}")

elapsed = time.time() - start_time
print(f"\n✅ Transcription completed in {elapsed/60:.1f} minutes!")

# Save JSON
with open(TRANSCRIPT_JSON, "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=2, ensure_ascii=False)

# Save readable TXT version
with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f:
    for seg in transcript:
        f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")

print(f"💾 Saved transcript to:")
print(f"   • {TRANSCRIPT_JSON}")
print(f"   • {TRANSCRIPT_TXT}")
