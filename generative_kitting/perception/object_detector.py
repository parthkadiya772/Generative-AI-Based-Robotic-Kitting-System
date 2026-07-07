"""
Zero-Shot Object Detector for the Generative Kitting System.

Provides precise pixel-level bounding boxes for detected objects using
a zero-shot detection model.  Unlike VLMs, which give rough coordinate
estimates (~15% pixel error), detection models return exact bounding
boxes from the vision encoder — no coordinate hallucination.

Used in the hybrid perception pipeline:
  VLM  → semantic labels, affordances, scene understanding
  Detector → precise bounding box coordinates (this module)

Backends:
  - **Remote** (default): calls ``detector_server.py`` running on the
    same GPU server as Ollama.  No local torch/transformers needed.
  - **Local**: loads OWL-ViT2 or Grounding DINO via HuggingFace
    Transformers directly in-process.
"""

import base64
import io
import json
import urllib.request
from typing import Any, Dict, List, Optional

from PIL import Image

from utils.logger import log


# Known part types in the kitting workspace
KNOWN_PART_LABELS = [
    "motor valve",
    "black hose",
    "black plate",
    "black plug",
    "small hinge",
    "small tube",
    "silver box",
    "silver gun",
    "tube with clamps",
    "gear",
    "large gear",
    "round gear"
    "white box",
    "kitting tray",
]


class ZeroShotDetector:
    """Zero-shot object detector — remote API or local inference.

    In remote mode (default), sends images to ``detector_server.py``
    running on the GPU server.  No local ML dependencies required.

    In local mode, loads OWL-ViT2 or Grounding DINO via HuggingFace
    Transformers in-process.
    """

    def __init__(self, config: dict = None):
        """
        Parameters
        ----------
        config : dict
            Optional configuration with keys:
            - ``detector_url``: URL of remote detector server
              (e.g. ``"http://10.7.0.35:8700"``).  If set, remote mode
              is used and no local model is loaded.
            - ``detector_model``: ``"owlv2"`` or ``"grounding_dino"``
            - ``detector_confidence``: min confidence (default 0.15)
            - ``detector_device``: ``"cpu"`` or ``"cuda"`` (local only)
        """
        cfg = config or {}
        self.detector_url = cfg.get("detector_url", "")
        self.model_name = cfg.get("detector_model", "owlv2")
        self.confidence_threshold = cfg.get("detector_confidence", 0.15)
        self.device = cfg.get("detector_device", "cpu")

        self._processor = None
        self._model = None
        self._available = None

        mode = "remote" if self.detector_url else "local"
        log.info(
            f"ZeroShotDetector init: mode={mode}, model={self.model_name}, "
            f"confidence={self.confidence_threshold}"
        )

    # ─── Public API ─────────────────────────────────────────

    def detect(
        self,
        image: Image.Image,
        queries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect objects in the image matching the text queries.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB image from the workspace camera.
        queries : list of str, optional
            Text descriptions to search for.  Defaults to KNOWN_PART_LABELS.

        Returns
        -------
        list of dict
            Each detection has:
            - ``label``: matched text query
            - ``confidence``: detection confidence (0-1)
            - ``bbox``: [x1, y1, x2, y2] in pixels
            - ``center``: normalised center {x, y} in [0, 1]
        """
        if not self.is_available:
            log.warning("ZeroShotDetector not available — returning empty")
            return []

        if queries is None:
            queries = KNOWN_PART_LABELS

        if self.detector_url:
            return self._detect_remote(image, queries)

        # Local inference
        if self.model_name == "owlv2":
            return self._detect_owlv2(image, queries)
        elif self.model_name == "grounding_dino":
            return self._detect_grounding_dino(image, queries)
        else:
            log.error(f"Unknown detector model: {self.model_name}")
            return []

    def detect_with_labels(
        self,
        image: Image.Image,
        vlm_labels: List[str],
    ) -> List[Dict[str, Any]]:
        """Detect objects using VLM-provided labels as queries.

        This is the primary integration point: the VLM identifies what
        parts are present, then this detector finds exactly where they
        are in pixels.
        """
        # Convert underscore labels to natural language for better matching
        queries = [label.replace("_", " ") for label in vlm_labels]
        queries_dedup = list(dict.fromkeys(queries + vlm_labels))
        return self.detect(image, queries_dedup)

    # ─── Remote Backend (HTTP) ──────────────────────────────

    def _detect_remote(
        self, image: Image.Image, queries: List[str],
    ) -> List[Dict[str, Any]]:
        """Send image to the remote detector server."""
        # Encode image as base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = json.dumps({
            "image_base64": b64_image,
            "queries": queries,
            "confidence": self.confidence_threshold,
        }).encode("utf-8")

        url = f"{self.detector_url.rstrip('/')}/detect"
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())

            detections = result.get("detections", [])
            inference_ms = result.get("inference_ms", 0)
            log.info(
                f"Remote detector: {len(detections)} objects in "
                f"{inference_ms:.0f}ms ({result.get('model', '?')})"
            )
            return detections

        except Exception as e:
            log.warning(f"Remote detector failed: {e}")
            return []

    # ─── OWL-ViT2 Local Backend ─────────────────────────────

    def _detect_owlv2(
        self, image: Image.Image, queries: List[str],
    ) -> List[Dict[str, Any]]:
        """Run zero-shot detection with OWL-ViT2 locally."""
        import torch

        if self._model is None:
            self._load_owlv2()

        w, h = image.size
        inputs = self._processor(
            text=[queries], images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs, threshold=self.confidence_threshold,
            target_sizes=[(h, w)], text_labels=[queries],
        )[0]

        detections = []
        for box, score, label_text in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["text_labels"],
        ):
            x1, y1, x2, y2 = box
            label = label_text if isinstance(label_text, str) else "unknown"
            detections.append({
                "label": label.replace(" ", "_"),
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": {
                    "x": float((x1 + x2) / 2.0 / w),
                    "y": float((y1 + y2) / 2.0 / h),
                },
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Deduplicate: keep highest-confidence per label, unless spatially separate
        seen = {}
        unique = []
        for det in detections:
            label = det["label"]
            if label not in seen:
                seen[label] = det
                unique.append(det)
            else:
                iou = self._compute_iou(seen[label]["bbox"], det["bbox"])
                if iou < 0.3:
                    unique.append(det)

        log.info(f"OWL-ViT2 (local): {len(unique)} detections")
        return unique

    def _load_owlv2(self):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        model_id = "google/owlv2-base-patch16-ensemble"
        log.info(f"Loading OWL-ViT2: {model_id} (device={self.device})")
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()
        log.info("OWL-ViT2 loaded")

    # ─── Grounding DINO Local Backend ───────────────────────

    def _detect_grounding_dino(
        self, image: Image.Image, queries: List[str],
    ) -> List[Dict[str, Any]]:
        """Run zero-shot detection with Grounding DINO locally."""
        import torch

        if self._model is None:
            self._load_grounding_dino()

        w, h = image.size
        text_prompt = ". ".join(queries) + "."
        inputs = self._processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=[(h, w)],
        )[0]

        detections = []
        for box, score, label_text in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append({
                "label": label_text.strip().replace(" ", "_"),
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": {
                    "x": float((x1 + x2) / 2.0 / w),
                    "y": float((y1 + y2) / 2.0 / h),
                },
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        log.info(f"Grounding DINO (local): {len(detections)} detections")
        return detections

    def _load_grounding_dino(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-tiny"
        log.info(f"Loading Grounding DINO: {model_id} (device={self.device})")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()
        log.info("Grounding DINO loaded")

    # ─── Utilities ──────────────────────────────────────────

    @staticmethod
    def _compute_iou(box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    @property
    def is_available(self) -> bool:
        """Check if the detector is reachable (remote) or importable (local)."""
        if self._available is not None:
            return self._available

        if self.detector_url:
            # Remote mode: check if server is reachable
            try:
                url = f"{self.detector_url.rstrip('/')}/health"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        self._available = True
                        log.info(
                            f"Remote detector available: {data.get('model')} "
                            f"on {data.get('device')}"
                        )
                        return True
            except Exception as e:
                log.warning(f"Remote detector not reachable at {self.detector_url}: {e}")
            self._available = False
        else:
            # Local mode: check if transformers + torch importable
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                log.warning(
                    "Zero-shot detector requires either:\n"
                    "  1. Remote: set detector_url in config.yaml\n"
                    "  2. Local: pip install transformers torch"
                )

        return self._available
