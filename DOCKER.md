# Running the app in Docker

The app is containerized so it runs identically anywhere Docker is installed —
no local Python/venv needed.

## Quick start (Docker Compose — recommended)

From the project root (`data_mining/`):

```bash
docker compose up --build
```

Then open **http://localhost:8501**.

Stop it with `Ctrl+C`, then clean up the container with:

```bash
docker compose down
```

## Quick start (plain Docker)

```bash
# Build the image (first time, or after changing code/deps)
docker build -t data-analysis-platform .

# Run it
docker run --rm -p 8501:8501 data-analysis-platform
```

Open **http://localhost:8501**. Stop with `Ctrl+C` (`--rm` auto-removes the container).

## Run in the background

```bash
docker run -d --name dap -p 8501:8501 data-analysis-platform
docker logs -f dap        # follow logs
docker stop dap && docker rm dap   # stop + remove
```

## Notes

- **Port:** the app listens on `8501` inside the container, published to `8501`
  on your machine. To use a different host port, change the mapping, e.g.
  `-p 9000:8501` → http://localhost:9000.
- **Health:** the container has a `HEALTHCHECK` hitting Streamlit's
  `/_stcore/health`; `docker ps` shows `healthy` once it's up (~25s).
- **Data is ephemeral:** uploaded files and saved dashboards live inside the
  container and reset when it's removed — same behaviour as the hosted demo.
- **Config:** `.streamlit/config.toml` (dark theme + 50 MB upload cap) is baked
  into the image, so the look and limits match local/cloud exactly.
- **Image size:** ~1.2 GB (the scientific stack — pandas/numpy/scipy/scikit-learn/
  matplotlib — is large). The build caches dependencies, so code-only changes
  rebuild in seconds.
