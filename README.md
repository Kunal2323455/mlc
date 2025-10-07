Tulsi Disease Detection – API + Frontend
=======================================

Quick start (Windows PowerShell)
--------------------------------

1) Create/activate venv (optional if you already use `tulsi_env`)

```powershell
python -m venv tulsi_env
./tulsi_env/Scripts/Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Run API server (serves frontend)

```powershell
python api.py
```

The server starts at `http://127.0.0.1:8000/`. The root serves the UI. The API exposes:

- GET `/health` – health check
- POST `/predict` – multipart form with `file` image

Frontend usage
--------------

1) Open `http://127.0.0.1:8000/` in your browser
2) Drag-and-drop or browse to select an image
3) Click Predict to see diagnosis, confidence, class probabilities, and treatment advice

Notes
-----

- The API loads `tulsi_disease_detection_best_model.h5` and `model_config.json` from the project root.
- For production, restrict CORS and consider running with `uvicorn --host 0.0.0.0 --port 8000 --workers 2 api:app`.
- If you need to call the API from another host or port, set `window.API_BASE` in `frontend/index.html` to that base URL.


