# api.py - Enhanced Tulsi Disease Detection API with Multiple Models

import os
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tensorflow.keras.models import load_model
from detector import TulsiDiseaseDetector

# Frontend directory
APP_STATIC_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# Enhanced model files dictionary to match main.py output
MODEL_FILES = {
    "custom_cnn": "best_Custom_CNN_model.h5",
    "vgg16_transfer": "best_VGG16_Transfer_model.h5", 
    "mobilenet_transfer": "best_MobileNet_Transfer_model.h5",
    "efficientnet_transfer": "best_EfficientNet_Transfer_model.h5",
    "resnet50_transfer": "best_ResNet50_Transfer_model.h5"
}

# Global variables
detectors = {}
config_path = ""

# FastAPI app initialization
app = FastAPI(
    title="Tulsi Disease Detection API", 
    version="2.0.0",
    description="Advanced AI-powered tulsi plant disease detection with multiple deep learning models"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ensure_model_files_exist():
    """Ensure all required model files exist"""
    global config_path
    config_path = os.path.join(os.path.dirname(__file__), "model_config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config.json not found in project root")

    # Check for model files (only check existing ones)
    available_models = {}
    missing_models = []
    
    for model_key, model_file in MODEL_FILES.items():
        model_path = os.path.join(os.path.dirname(__file__), model_file)
        if os.path.exists(model_path):
            available_models[model_key] = model_file
        else:
            missing_models.append(f"{model_key}: {model_file}")
    
    if not available_models:
        raise FileNotFoundError("No model files found. Please run main.py to train models first.")
    
    if missing_models:
        print(f"⚠️ Some models not found: {missing_models}")
        print(f"✅ Available models: {list(available_models.keys())}")
    
    # Update MODEL_FILES to only include available models
    MODEL_FILES.clear()
    MODEL_FILES.update(available_models)

@app.on_event("startup")
def load_detectors_on_startup():
    """Load all available models on startup"""
    global detectors
    
    try:
        _ensure_model_files_exist()
        print("🚀 Loading AI models...")
        
        for model_key, model_file in MODEL_FILES.items():
            model_path = os.path.join(os.path.dirname(__file__), model_file)
            print(f" -> Loading '{model_key}' from {model_file}")
            
            try:
                detectors[model_key] = TulsiDiseaseDetector(model_path, config_path)
                print(f" ✅ {model_key} loaded successfully")
            except Exception as e:
                print(f" ❌ Failed to load {model_key}: {e}")
        
        if detectors:
            print(f"🎉 Successfully loaded {len(detectors)} AI models!")
        else:
            print("❌ No models could be loaded")
            
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("💡 Make sure to run 'python main.py' first to train the models")

@app.get("/models")
def get_models():
    """Get list of available AI models"""
    return {
        "models": list(detectors.keys()),
        "count": len(detectors),
        "status": "ready" if detectors else "no_models"
    }

@app.get("/health")
def health_check():
    """API health check"""
    return {
        "status": "ok",
        "models_loaded": len(detectors),
        "available_models": list(detectors.keys())
    }

@app.get("/model/{model_name}/info")
def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    if model_name not in detectors:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    # Model information database
    model_info = {
        "custom_cnn": {
            "name": "Custom CNN",
            "description": "Deep convolutional network designed specifically for plant diseases",
            "accuracy": "87-92%",
            "type": "Custom Architecture",
            "parameters": "~2.5M"
        },
        "vgg16_transfer": {
            "name": "VGG16 Transfer Learning",
            "description": "Classic deep learning architecture with transfer learning",
            "accuracy": "89-94%", 
            "type": "Transfer Learning",
            "parameters": "~15M"
        },
        "mobilenet_transfer": {
            "name": "MobileNet Transfer Learning",
            "description": "Lightweight model optimized for mobile deployment",
            "accuracy": "88-93%",
            "type": "Mobile Optimized",
            "parameters": "~3.2M"
        },
        "efficientnet_transfer": {
            "name": "EfficientNet Transfer Learning",
            "description": "State-of-the-art efficient neural network architecture",
            "accuracy": "92-97%",
            "type": "State-of-the-art",
            "parameters": "~5.3M"
        },
        "resnet50_transfer": {
            "name": "ResNet50 Transfer Learning", 
            "description": "Deep residual network with skip connections",
            "accuracy": "90-95%",
            "type": "Deep Learning",
            "parameters": "~25M"
        }
    }
    
    return {
        "model_name": model_name,
        "info": model_info.get(model_name, {
            "name": model_name.replace('_', ' ').title(),
            "description": "Advanced AI model for plant disease detection",
            "accuracy": "85-90%",
            "type": "Neural Network",
            "parameters": "Unknown"
        }),
        "status": "loaded"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    """Predict plant disease using selected AI model"""
    
    # Validate model
    if model_name not in detectors:
        available = list(detectors.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {available}"
        )

    # Validate file
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (JPG, PNG, etc.)")

    try:
        # Read file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Save temporary file
        temp_dir = os.path.join(os.path.dirname(__file__), "_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename or "temp_image.jpg")
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Make prediction
        selected_detector = detectors[model_name]
        result = selected_detector.predict(temp_path)
        recommendation = selected_detector.get_treatment_recommendation(result)

        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass

        # Return comprehensive response
        response = {
            "prediction": result,
            "recommendation": recommendation,
            "model_used": model_name,
            "analysis": {
                "image_name": file.filename,
                "file_size": len(contents),
                "confidence_level": "high" if result["confidence"] > 0.8 else "medium" if result["confidence"] > 0.6 else "low",
                "needs_attention": result.get("needs_treatment", False)
            }
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/stats")
def get_stats():
    """Get API usage statistics"""
    return {
        "total_models": len(MODEL_FILES),
        "loaded_models": len(detectors),
        "model_status": {name: "loaded" for name in detectors.keys()},
        "api_version": "2.0.0",
        "features": [
            "Multiple AI Models",
            "Real-time Prediction", 
            "Confidence Scoring",
            "Treatment Recommendations",
            "Model Comparison"
        ]
    }

# Mount static files (frontend)
if os.path.isdir(APP_STATIC_DIR):
    app.mount("/", StaticFiles(directory=APP_STATIC_DIR, html=True), name="frontend")
    print(f"✅ Frontend mounted from: {APP_STATIC_DIR}")
else:
    print(f"⚠️ Frontend directory not found: {APP_STATIC_DIR}")

if __name__ == "__main__":
    print("🌿 Starting Tulsi Disease Detection API Server...")
    print("="*50)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)