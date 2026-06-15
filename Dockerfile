# Data Analysis Platform — containerized Streamlit app
# Build:  docker build -t data-analysis-platform .
# Run:    docker run --rm -p 8501:8501 data-analysis-platform
# Open:   http://localhost:8501

FROM python:3.11-slim

# Runtime system libs only:
#   libgomp1 — OpenMP runtime needed by scikit-learn / scipy
#   curl     — used by the container HEALTHCHECK
# (all Python deps ship manylinux wheels for linux/amd64, so no compiler needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached unless requirements change
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the Streamlit config (theme + upload cap)
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Liveness probe against Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Launch from the repo root so .streamlit/config.toml and the __file__-anchored
# static/upload paths resolve exactly as they do locally and on Streamlit Cloud.
ENTRYPOINT ["streamlit", "run", "app/home.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
