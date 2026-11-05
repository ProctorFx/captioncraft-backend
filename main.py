
import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI(title="CaptionCraft AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Chinese": "zh",
    "Korean": "ko"
}


def make_srt(segments):
    result = []
    for i, seg in enumerate(segments, 1):
        start = float(seg["start"])
        end = float(seg["end"])

        def timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

        result.append(
            f"{i}\n{timestamp(start)} --> {timestamp(end)}\n{seg['text']}\n"
        )

    return "\n".join(result)


def make_vtt(segments):
    result = ["WEBVTT\n"]

    for seg in segments:
        def timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

        result.append(
            f"{timestamp(float(seg['start']))} --> "
            f"{timestamp(float(seg['end']))}\n{seg['text']}\n"
        )

    return "\n".join(result)


@app.get("/")
def home():
    return {"message": "CaptionCraft AI Server is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/captions")
async def generate_captions(
    video: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str = Form("auto")
):
    if not client.api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server."
        )

    if target_language not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported target language."
        )

    allowed_extensions = {
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg"
    }

    extension = Path(video.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Please upload a supported video file."
        )

    job_id = str(uuid.uuid4())
    input_path = BASE_DIR / f"{job_id}{extension}"
    srt_path = OUTPUT_DIR / f"{job_id}.srt"
    vtt_path = OUTPUT_DIR / f"{job_id}.vtt"
    mp4_path = OUTPUT_DIR / f"{job_id}.mp4"

    try:
        with open(input_path, "wb") as output:
            shutil.copyfileobj(video.file, output)

        # Extract audio and transcribe it with Whisper.
        with open(input_path, "rb") as audio:
            options = {
                "model": "whisper-1",
                "file": audio,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"]
            }

            if source_language != "auto":
                options["language"] = LANGUAGES.get(
                    source_language, source_language
                )

            transcript = client.audio.transcriptions.create(**options)

        original_segments = transcript.segments or []

        if not original_segments:
            raise HTTPException(
                status_code=422,
                detail="No speech segments were detected in the video."
            )

        # Prepare timestamped text for translation.
        source_data = [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip()
            }
            for seg in original_segments
        ]

        # Translate the captions while preserving their timestamps.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate each caption into the requested language. "
                        "Preserve the meaning and timestamps. "
                        "Return valid JSON with a segments array. "
                        "Each item must contain start, end and text."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Target language: {target_language}\n"
                        f"Captions: {json.dumps(source_data, ensure_ascii=False)}"
                    )
                }
            ]
        )

        translated = json.loads(
            response.choices[0].message.content
        )

        segments = translated.get("segments", [])

        if len(segments) != len(source_data):
            raise ValueError("Translation returned an invalid number of captions.")

        # Generate subtitle files.
        srt_content = make_srt(segments)
        vtt_content = make_vtt(segments)

        srt_path.write_text(srt_content, encoding="utf-8")
        vtt_path.write_text(vtt_content, encoding="utf-8")

        # Burn subtitles into the video if FFmpeg is available.
        captioned_video_url = None

        if shutil.which("ffmpeg"):
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-vf", f"subtitles={srt_path}",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(mp4_path)
                ],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and mp4_path.exists():
                captioned_video_url = f"/outputs/{job_id}.mp4"

        return {
            "detected_language": getattr(
                transcript, "language", "unknown"
            ),
            "target_language": target_language,
            "segments": segments,
            "srt": srt_content,
            "vtt": vtt_content,
            "captioned_video_url": captioned_video_url
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Caption generation failed: {str(error)}"
        )

    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)
