from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import cv2
import numpy as np
import base64

from app.models.capsule_resnet import Model1
from app.models.dedswin import Model2
from app.utils.image_proc import load_medical_image, localize_tumor, create_segmentation_overlay, simulate_metrics

app = FastAPI(title="Chronic Liver Disease Detection System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session data for demo
# In production, use Redis or Postgres
STORAGE_DIR = "data/processed/temp"
os.makedirs(STORAGE_DIR, exist_ok=True)

class DiagnosisResult(BaseModel):
    id: str
    model: str
    disease: str
    tumor_location: str
    metrics: dict
    image_base64: str

@app.post("/analyze", response_model=DiagnosisResult)
async def analyze_image(file: UploadFile = File(...), model_id: int = 1):
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    input_path = os.path.join(STORAGE_DIR, f"{file_id}.{ext}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Load and Preprocess
        img = load_medical_image(input_path)
        
        # 2. Model Inference (Simulated for this demo with real architectures)
        if model_id == 1:
            # Model 1 logic: ResNet-Capsule
            disease_classes = ["Inflammation", "Fibrosis", "Cirrhosis"]
            disease = np.random.choice(disease_classes)
            
            # Simulated masks based on image shape
            liver_mask = np.zeros_like(img)
            cv2.circle(liver_mask, (128, 128), 70, 1, -1)
            
            tumor_mask = np.zeros_like(img)
            cv2.circle(tumor_mask, (140, 110), 15, 1, -1)
            
            metrics = simulate_metrics(1)
            model_name = "Capsule-ResNet"
            
        else:
            # Model 2 logic: DEDSWIN-Net
            disease_classes = ["Cirrhosis", "Hepatoma", "Hepatitis"]
            disease = "Chronic Liver Disease (Multi-scale Analysis)"
            
            liver_mask = np.zeros_like(img)
            cv2.circle(liver_mask, (128, 128), 80, 1, -1)
            
            tumor_mask = np.zeros_like(img)
            cv2.circle(tumor_mask, (110, 140), 20, 1, -1)
            
            metrics = simulate_metrics(2)
            model_name = "DEDSWIN-Net"
            
        # 3. Tumor Localization
        loc = localize_tumor(liver_mask, tumor_mask)
        
        # 4. Create Visualization Overlay
        overlay = create_segmentation_overlay(img, liver_mask, tumor_mask)
        
        # Encode for frontend
        _, buffer = cv2.imencode('.jpg', overlay)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return DiagnosisResult(
            id=file_id,
            model=model_name,
            disease=disease,
            tumor_location=loc,
            metrics=metrics,
            image_base64=f"data:image/jpeg;base64,{img_str}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
