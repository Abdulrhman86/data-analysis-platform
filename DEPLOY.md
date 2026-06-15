# Deploying to Streamlit Community Cloud (free, ~10 minutes)

The app is already deploy-ready (pinned `requirements.txt`, `runtime.txt`,
`.streamlit/config.toml`, no machine-specific paths). You only need to do the
two account steps below.

## Prerequisites
- A **GitHub** account (free).
- A **Streamlit Community Cloud** account (free — sign in with GitHub):
  <https://share.streamlit.io>

## 1. Push the repo to GitHub
Create an empty repo on github.com (e.g. `data-analysis-platform`, no README),
then from the project root (`data_mining/`):

```bash
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

The repo already has 13+ commits and everything Cloud needs.
(If you'd rather I push for you, install + authenticate the GitHub CLI
`gh auth login`, then tell me — I can create the remote and push.)

## 2. Deploy on Streamlit Community Cloud
1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Pick your repo and the `main` branch.
4. Set **Main file path** to exactly: `app/home.py`
5. Click **Deploy**. The first build takes a few minutes (it installs
   `requirements.txt`). You'll get a public URL like
   `https://<your-app>.streamlit.app`.

## 3. Verify it works
Open the URL and click through: **Get Started → Try sample data → Data Quality →
Preprocessing → Visualization → Dashboard → Machine Learning** (train + predict).
(Tell me the URL and I can drive it end-to-end to confirm.)

## Notes & limits (already handled in the app)
- **Uploads capped at 50 MB** (`.streamlit/config.toml`) to stay within Cloud's
  ~1 GB memory; the data-quality engine samples very large datasets.
- **Dashboards** persist to disk, but Cloud's filesystem is **ephemeral** — they
  may reset on restart. Use the dashboard **Import/Export** to keep a durable copy.
- Don't commit secrets; use Cloud's **Secrets** manager if you ever need any.
- Python is pinned to **3.11** via `runtime.txt`.

## Updating the live app
Just `git push` to `main` — Streamlit Cloud auto-redeploys.
