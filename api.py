# api.py - Tulsi Disease Detection API

import os
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import TulsiDiseaseDetector

APP_STATIC_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# Single model configuration
MODEL_FILE = "tulsi_disease_detection_best_model.h5"
detector = None
config_path = ""

app = FastAPI(title="Tulsi Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_model_files_exist():
    global config_path
    config_path = os.path.join(os.path.dirname(__file__), "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config.json not found in project root")

    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{MODEL_FILE} not found")


@app.on_event("startup")
def load_detector_on_startup():
    global detector
    _ensure_model_files_exist()
    print("Loading model...")
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    print(f" -> Loading model from {MODEL_FILE}")
    detector = TulsiDiseaseDetector(model_path, config_path)
    print("Model loaded successfully!")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        temp_dir = os.path.join(os.path.dirname(__file__), "_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Make prediction
        result = detector.predict(temp_path)
        recommendation = detector.get_treatment_recommendation(result)

        response = {
            "prediction": result,
            "recommendation": recommendation,
            "model_used": "Tulsi Disease Detection CNN"
        }
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def get_model_info():
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": "Tulsi Disease Detection CNN",
        "classes": detector.class_names,
        "input_size": f"{detector.img_height}x{detector.img_width}",
        "description": "Deep learning model for detecting diseases in Tulsi plants"
    }


if os.path.isdir(APP_STATIC_DIR):
    app.mount("/", StaticFiles(directory=APP_STATIC_DIR, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)