from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from infernce import BrainTumorInferenceService


inference_service: BrainTumorInferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service
    inference_service = BrainTumorInferenceService()
    yield
    inference_service = None


app = FastAPI(
    title="Brain Tumor Inference API",
    version="1.0.0",
    description="FastAPI backend for the original DenseNet121 model and the mobile Lite export.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Brain Tumor Inference API is running"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "models_loaded": inference_service is not None,
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict[str, object]:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Models are still loading")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        predictions = inference_service.predict_from_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}") from exc

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        **predictions,
    }
