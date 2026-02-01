# 🚀 Deployment Guide

You have two powerful ways to deploy **RentShield Germany**.

## Option 1: The "Split" Architecture (Standard)
*Use this if you want a separate API that other apps can also use.*

- **Backend (FastAPI)** → Deployed on Render / Fly.io
- **Frontend (Streamlit)** → Deployed on Streamlit Community Cloud

**Steps:**
1. Deploy `backend` folder to Render.
2. Set environment variable `RENTSHIELD_API_URL` in Streamlit Cloud to your Render URL.
3. Deploy `frontend/app.py` to Streamlit Cloud.

---

## Option 2: The "Bundled" Architecture (Recommended for You)
*Use this to run EVERYTHING on Streamlit Community Cloud without needing a separate backend server.*

Since you want to run the FastAPI logic "on Streamlit", we use a **Bundled App**. This imports the backend logic directly into Streamlit, bypassing the need for a separate network server.

### 1. The Bundled File
We created a special file: **`frontend/bundled_app.py`**
This file:
- Loads your ML models directly from `backend/app/artifacts`
- Runs the prediction logic internally
- Does NOT require `uvicorn` or `fastapi` to be running
- Works perfectly on the free Streamlit Community Cloud

### 2. How to Deploy on Streamlit Cloud
1. Push your code to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository.
4. **Main File Path:** Set this to `frontend/bundled_app.py`
5. Click **Deploy!**

### 3. Dependencies
Ensure your root `requirements.txt` includes everything needed for both:
```text
streamlit>=1.28.0
plotly>=5.18.0
scikit-learn>=1.3.0
pandas>=2.1.0
joblib>=1.3.0
# fastapi/uvicorn are technically not needed for this mode, but safe to keep
```

Done! You now have the full ML power running entirely on Streamlit.
