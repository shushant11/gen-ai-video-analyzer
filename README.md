# 🎬 Gen-AI Video Analyzer

**Transcribe · Summarise · Chat with your meetings and videos**

Gen-AI Video Analyzer turns any YouTube video or local audio/video file into structured meeting intelligence. It transcribes the audio, generates a title and summary, pulls out action items, key decisions and open questions, and then lets you **chat with the content** through a Retrieval-Augmented Generation (RAG) pipeline.

It supports both **English** (via local OpenAI Whisper) and **Hinglish / Hindi** (via Sarvam AI speech-to-text-translate) audio.

---

## ✨ Features

- **Flexible input** – paste a YouTube URL or a path to a local audio/video file (`.mp4`, `.mp3`, `.wav`, `.m4a`, etc.)
- **Dual transcription engines**
  - `english` → OpenAI Whisper, running locally
  - `hinglish` → Sarvam AI (`saaras:v2.5`), which transcribes and translates to English in one step
- **Automatic title** – a short professional title (max 8 words)
- **Map-reduce summarisation** – long transcripts are split, summarised chunk by chunk, then merged into one bullet-point summary
- **Smart extraction**
  - ✅ Action items with owner and deadline
  - 🔑 Key decisions
  - ❓ Open questions / follow-ups
- **RAG chat** – ask questions about the video; answers come only from the transcript
- **Two interfaces** – a styled Streamlit web app and an interactive command-line mode

---

## 🧠 How It Works

```
 YouTube URL / Local File
            │
            ▼
 ┌────────────────────────┐
 │  Audio Processing      │  yt-dlp download → WAV (16 kHz mono) → 10-min chunks
 └────────────────────────┘
            │
            ▼
 ┌────────────────────────┐
 │  Transcription         │  Whisper (English)  |  Sarvam AI (Hinglish → English)
 └────────────────────────┘
            │
            ▼
 ┌────────────────────────┐
 │  LLM Analysis          │  Mistral (mistral-small-latest) via LangChain LCEL
 │                        │  → Title · Summary · Action Items · Decisions · Questions
 └────────────────────────┘
            │
            ▼
 ┌────────────────────────┐
 │  RAG Engine            │  500-char chunks → all-MiniLM-L6-v2 embeddings → ChromaDB
 │                        │  → top-4 retrieval → Mistral answers your questions
 └────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Audio acquisition | `yt-dlp`, `pydub`, FFmpeg |
| Speech-to-text | OpenAI Whisper (local), Sarvam AI API |
| LLM | Mistral AI (`mistral-small-latest`) |
| Orchestration | LangChain (LCEL) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| UI | Streamlit |

---

## 📁 Project Structure

```
gen-ai-video-analyzer/
├── app.py                  # Streamlit web interface
├── main.py                 # CLI entry point + run_pipeline()
├── test.py                 # Quick end-to-end test script
├── Requirements.txt        # Python dependencies
├── core/
│   ├── transcriber.py      # Whisper + Sarvam AI transcription
│   ├── summarizer.py       # Map-reduce summary + title generation
│   ├── extractor.py        # Action items, decisions, open questions
│   ├── vector_store.py     # ChromaDB + HuggingFace embeddings
│   └── rag_engine.py       # RAG chain for Q&A
└── utils/
    └── audio_processor.py  # YouTube download, WAV conversion, chunking
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+**
- **FFmpeg** installed on your system ([download](https://ffmpeg.org/download.html))
- A **Mistral AI API key** ([console.mistral.ai](https://console.mistral.ai/))
- A **Sarvam AI API key** (only needed for Hinglish) ([sarvam.ai](https://www.sarvam.ai/))

### 2. Clone the repository

```bash
git clone https://github.com/shushant11/gen-ai-video-analyzer.git
cd gen-ai-video-analyzer
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r Requirements.txt
```

### 5. Configure FFmpeg

If FFmpeg is on your system `PATH` (check with `ffmpeg -version`), nothing else is needed.
Otherwise add its full path to `.env`:

```env
FFMPEG_PATH=C:\\ffmpeg\\bin\\ffmpeg.exe
```

### 6. Set up environment variables

Copy `.env.example` to `.env` in the project root and fill in your keys:

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key

# Optional
WHISPER_MODEL=small            # tiny | base | small | medium | large
SARVAM_STT_MODEL=saaras:v2.5
```

---

## 💻 Usage

### Streamlit web app

```bash
streamlit run app.py
```

1. Paste a YouTube URL or a local file path in the sidebar
2. Choose the language: `english` or `hinglish`
3. Click **⚡ Analyse**
4. Review the title, summary, transcript, action items, decisions and open questions
5. Use **💬 Chat with your Meeting** to ask follow-up questions

### Command line

```bash
python main.py
```

```
Enter YouTube URL or local file path: https://www.youtube.com/watch?v=...
Language (english/hinglish): english
...
💬 Chat with your meeting (type 'exit' to quit)
You: What were the main decisions made?
```

### Quick test

Edit the `source` and `language` values in `test.py`, then run:

```bash
python test.py
```

---

## ⚙️ Configuration Notes

| Setting | Where | Default |
|---|---|---|
| Whisper model size | `WHISPER_MODEL` env var | `small` |
| Audio chunk length | `chunk_audio()` in `utils/audio_processor.py` | 10 minutes |
| Sarvam piece length | `SARVAM_PIECE_SECONDS` in `core/transcriber.py` | 25 seconds (API limit is 30 s) |
| Summary chunk size | `split_transcript()` in `core/summarizer.py` | 3000 chars, 200 overlap |
| RAG chunk size | `build_vector_store()` in `core/vector_store.py` | 500 chars, 50 overlap |
| Retrieved chunks (k) | `build_rag_chain()` in `core/rag_engine.py` | 4 |
| Vector DB location | `CHROMA_DIR` in `core/vector_store.py` | `vector_db/` |

Downloaded audio is saved to the `downloades/` folder (git-ignored).

---

## 🗺️ Roadmap

- [ ] Export results to PDF / TXT (`reportlab` and `fpdf2` are already in the requirements)
- [ ] Speaker diarization
- [ ] Remove the hard-coded FFmpeg path (auto-detect or read from `.env`)
- [ ] Per-video vector store collections so multiple videos don't mix
- [ ] Docker support

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, commit your changes and open a pull request.

```bash
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

---

## 📄 License

No license has been added yet. Consider adding one (for example MIT) so others know how they can use this project.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Sarvam AI](https://www.sarvam.ai/)
- [Mistral AI](https://mistral.ai/)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
