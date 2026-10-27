import os
import shutil

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"

# FFmpeg location:
#   - If FFMPEG_PATH is set in .env (full path to ffmpeg / ffmpeg.exe), use it.
#   - Otherwise FFmpeg must be on the system PATH.
FFMPEG_PATH = os.getenv("FFMPEG_PATH")

if FFMPEG_PATH:
    AudioSegment.converter = FFMPEG_PATH
elif shutil.which("ffmpeg") is None:
    print("⚠️  FFmpeg not found on PATH. Install it or set FFMPEG_PATH in .env")


def download_youtube_audio(url: str) -> str:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    if FFMPEG_PATH:
        ydl_opts["ffmpeg_location"] = os.path.dirname(FFMPEG_PATH)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Path of the file after the WAV post-processor has run
        downloads = info.get("requested_downloads") or []
        if downloads and downloads[0].get("filepath"):
            filename = downloads[0]["filepath"]
        else:
            filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Downloaded audio not found: {filename}")
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV using pydub."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    source = source.strip().strip('"')

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created")
    return chunks
