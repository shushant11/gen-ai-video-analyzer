# Hosted demo image (Hugging Face Spaces, Docker SDK). Runs the same app with HOSTED_DEMO=1.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HOSTED_DEMO=1 \
    WHISPER_MODEL=base \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch first (much smaller than the default CUDA build), then everything else
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY --chown=user . .

# Download the models at build time so the first visitor does not wait
RUN python -c "import whisper; whisper.load_model('base')" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860
# XSRF/CORS are off because Spaces serve the app inside an iframe, where file uploads otherwise fail with 403
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--server.maxUploadSize=100", \
     "--browser.gatherUsageStats=false"]
