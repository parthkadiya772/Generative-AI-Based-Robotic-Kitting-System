"""
Zero-Shot Object Detection Server.

Run this on the remote GPU server (same machine as Ollama) to serve
OWL-ViT2 or Grounding DINO detections over HTTP.

Usage:
    pip install transformers torch fastapi uvicorn pillow
    python detector_server.py                          # default: owlv2, port 8700
    python detector_server.py --model grounding_dino   # use Grounding DINO
    python detector_server.py --port 8700 --device cuda # explicit GPU

API:
    POST /detect
      Body: { "image_base64": "<base64 PNG/JPEG>",
              "queries": ["motor valve", "black hose", ...],
              "confidence": 0.15 }
      Returns: { "detections": [ {label, confidence, bbox, center}, ... ] }

    GET /health
      Returns: { "status": "ok", "model": "owlv2", "device": "cuda" }
"""

import argparse
import base64
import io
import os
import time

import numpy as np
import torch
from PIL import Image

# ─── Model Loading ──────────────────────────────────────────

MODEL = None
PROCESSOR = None
MODEL_NAME = "owlv2"
DEVICE = "cpu"


def load_owlv2(device: str):
    global MODEL, PROCESSOR, DEVICE
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    model_id = "google/owlv2-base-patch16-ensemble"
    print(f"Loading OWL-ViT2: {model_id} on {device}...")
    PROCESSOR = Owlv2Processor.from_pretrained(model_id)
    MODEL = Owlv2ForObjectDetection.from_pretrained(model_id)
    MODEL.to(device)
    MODEL.eval()
    DEVICE = device
    print("OWL-ViT2 ready.")


def load_grounding_dino(device: str):
    global MODEL, PROCESSOR, DEVICE
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading Grounding DINO: {model_id} on {device}...")
    PROCESSOR = AutoProcessor.from_pretrained(model_id)
    MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    MODEL.to(device)
    MODEL.eval()
    DEVICE = device
    print("Grounding DINO ready.")


# ─── Detection Functions ────────────────────────────────────

def detect_owlv2(image: Image.Image, queries: list, confidence: float) -> list:
    w, h = image.size
    inputs = PROCESSOR(text=[queries], images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    target_sizes = torch.tensor([[h, w]]).to(DEVICE)
    results = PROCESSOR.post_process_object_detection(
        outputs, threshold=confidence, target_sizes=target_sizes
    )[0]

    detections = []
    for box, score, label_idx in zip(
        results["boxes"].cpu().numpy(),
        results["scores"].cpu().numpy(),
        results["labels"].cpu().numpy(),
    ):
        x1, y1, x2, y2 = box
        label = queries[label_idx] if label_idx < len(queries) else "unknown"
        detections.append({
            "label": label.replace(" ", "_"),
            "confidence": round(float(score), 4),
            "bbox": [round(float(v), 1) for v in [x1, y1, x2, y2]],
            "center": {
                "x": round(float((x1 + x2) / 2.0 / w), 4),
                "y": round(float((y1 + y2) / 2.0 / h), 4),
            },
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def detect_grounding_dino(image: Image.Image, queries: list, confidence: float) -> list:
    w, h = image.size
    text_prompt = ". ".join(queries) + "."
    inputs = PROCESSOR(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    results = PROCESSOR.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=confidence,
        text_threshold=confidence,
        target_sizes=[(h, w)],
    )[0]

    detections = []
    for box, score, label_text in zip(
        results["boxes"], results["scores"], results["labels"]
    ):
        x1, y1, x2, y2 = box.cpu().numpy()
        detections.append({
            "label": label_text.strip().replace(" ", "_"),
            "confidence": round(float(score), 4),
            "bbox": [round(float(v), 1) for v in [x1, y1, x2, y2]],
            "center": {
                "x": round(float((x1 + x2) / 2.0 / w), 4),
                "y": round(float((y1 + y2) / 2.0 / h), 4),
            },
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ─── FastAPI Server ─────────────────────────────────────────

def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Zero-Shot Detector API")

    class DetectRequest(BaseModel):
        image_base64: str
        queries: Optional[List[str]] = None
        confidence: Optional[float] = 0.15

    class DetectResponse(BaseModel):
        detections: list
        model: str
        inference_ms: float

    DEFAULT_QUERIES = [
        "motor valve", "black hose", "black plate", "black plug",
        "small hinge", "small tube", "silver box", "silver gun",
        "tube with clamps", "white box", "kitting tray",
    ]

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        img_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        queries = req.queries or DEFAULT_QUERIES
        confidence = req.confidence or 0.15

        t0 = time.time()
        if MODEL_NAME == "owlv2":
            dets = detect_owlv2(image, queries, confidence)
        else:
            dets = detect_grounding_dino(image, queries, confidence)
        elapsed = (time.time() - t0) * 1000

        print(f"[detect] {len(dets)} objects in {elapsed:.0f}ms "
              f"({len(queries)} queries, conf>={confidence})")
        for d in dets:
            print(f"  {d['label']}: ({d['center']['x']:.3f}, {d['center']['y']:.3f}) "
                  f"conf={d['confidence']:.3f}")

        return DetectResponse(
            detections=dets, model=MODEL_NAME, inference_ms=round(elapsed, 1)
        )

    return app


# ─── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Detector Server")
    parser.add_argument("--model", default="owlv2", choices=["owlv2", "grounding_dino"])
    parser.add_argument("--port", type=int, default=8700)
    # Loopback by default — the detector endpoint has no auth and
    # exposes /detect to whoever can reach the port. Pass --host
    # 0.0.0.0 explicitly (or set DETECTOR_HOST in your environment)
    # when you knowingly want LAN access.
    parser.add_argument(
        "--host",
        default=os.environ.get("DETECTOR_HOST", "127.0.0.1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    MODEL_NAME = args.model

    if args.model == "owlv2":
        load_owlv2(args.device)
    else:
        load_grounding_dino(args.device)

    import uvicorn
    print(f"\nDetector server starting on {args.host}:{args.port}")
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device}")
    print(f"  POST /detect  — send image + queries")
    print(f"  GET  /health  — health check\n")
    uvicorn.run(create_app(), host=args.host, port=args.port)
