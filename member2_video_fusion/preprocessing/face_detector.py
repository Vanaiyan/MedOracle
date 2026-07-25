"""
member2_video_fusion/preprocessing/face_detector.py
=====================================================
MedOracle — Member 2 (Vanaiyan)

Face detection and cropping pipeline for CREMA-D video frames.

Primary detector  : YOLOv8 (ultralytics) — used when installed
Fallback detector : OpenCV Haar cascade   — used in dev/test environments

Pipeline per clip
-----------------
    raw frames (T, H, W, 3)
        → detect face bounding box in each frame
        → crop face region (with padding margin)
        → resize crop to 224 × 224
        → assess signal quality across T frames
    output: cropped frames (T, 224, 224, 3) + quality string

Signal quality thresholds (CLAUDE.md)
--------------------------------------
    Metric               Good        Degraded      Poor
    ------------------   ---------   -----------   ------
    Face detection rate  ≥ 80%       50 – 80%      < 50%
    Laplacian variance   ≥ 100       50 – 100      < 50
    Face bounding box    > 10% area  5 – 10%       < 5%

    Rule: worst-performing metric determines the overall grade.
    Citation: Pech-Pacheco et al. (2000) for Laplacian variance.

YOLO model note
---------------
    The recommended model is yolov8n-face.pt — a YOLOv8-nano model
    fine-tuned specifically for face detection.

    On first use, download it:
        from ultralytics import YOLO
        model = YOLO("yolov8n-face.pt")   # auto-downloads ~6 MB

    Or place it manually at:  member2_video_fusion/weights/yolov8n-face.pt

Author: Vanaiyan Kirupagaran (214215H)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_SIZE      = 224       # ResNet50 input size
FACE_PAD        = 0.20      # padding added around face crop (fraction of bbox size)
YOLO_CONF       = 0.40      # minimum YOLO confidence threshold
YOLO_MODEL_NAME = "yolov8n-face.pt"

# Quality thresholds
_DETECT_RATE_GOOD     = 0.80
_DETECT_RATE_DEGRADED = 0.50
_LAP_VAR_GOOD         = 100.0
_LAP_VAR_DEGRADED     = 50.0
_FACE_AREA_GOOD       = 0.10
_FACE_AREA_DEGRADED   = 0.05

SignalQuality = str   # "good" | "degraded" | "poor"


# ---------------------------------------------------------------------------
# Detector backend
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Face detector that uses YOLOv8 when available, Haar cascade otherwise.

    Parameters
    ----------
    model_path : path to yolov8n-face.pt (None → auto-download via ultralytics)
    conf       : YOLO confidence threshold (default 0.40)
    use_yolo   : force YOLO=True or YOLO=False; None → auto-detect

    Usage
    -----
        detector = FaceDetector()
        boxes = detector.detect(frame_rgb)   # returns list of (x1,y1,x2,y2)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf:       float         = YOLO_CONF,
        use_yolo:   Optional[bool] = None,
    ):
        self.conf    = conf
        self._yolo   = None
        self._haar   = None
        self._backend = "none"

        # Decide backend
        if use_yolo is True:
            self._init_yolo(model_path)
        elif use_yolo is False:
            self._init_haar()
        else:
            # Auto: try YOLO, fall back to Haar
            try:
                self._init_yolo(model_path)
            except Exception as e:
                logger.warning(f"YOLO unavailable ({e}), falling back to Haar cascade")
                self._init_haar()

        logger.info(f"FaceDetector initialised with backend: {self._backend}")

    # ── backend initialisers ────────────────────────────────────────────────

    def _init_yolo(self, model_path: Optional[str]) -> None:
        from ultralytics import YOLO  # ImportError propagates if not installed

        if model_path is None:
            # Check local weights folder first, then auto-download
            local = Path(__file__).parent.parent / "weights" / YOLO_MODEL_NAME
            model_path = str(local) if local.exists() else YOLO_MODEL_NAME

        self._yolo    = YOLO(model_path)
        self._backend = "yolo"

    def _init_haar(self) -> None:
        cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar    = cv2.CascadeClassifier(cascade_path)
        if self._haar.empty():
            raise RuntimeError("Haar cascade XML not found in OpenCV data directory")
        self._backend = "haar"

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Returns "yolo" or "haar"."""
        return self._backend

    def detect(self, frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a single RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray of shape (H, W, 3) uint8 in RGB

        Returns
        -------
        List of (x1, y1, x2, y2) bounding boxes in pixel coordinates.
        Empty list if no face detected.
        Boxes are sorted by area descending (largest face first).
        """
        if self._backend == "yolo":
            return self._detect_yolo(frame_rgb)
        else:
            return self._detect_haar(frame_rgb)

    # ── backend-specific detection ──────────────────────────────────────────

    def _detect_yolo(self, frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        results = self._yolo(frame_rgb, conf=self.conf, verbose=False)
        boxes   = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append((x1, y1, x2, y2))
        # Sort by area descending — largest face first
        boxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        return boxes

    def _detect_haar(self, frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return []
        # Convert (x, y, w, h) → (x1, y1, x2, y2), sort by area descending
        boxes = [(x, y, x+w, y+h) for x, y, w, h in faces]
        boxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        return boxes


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def crop_face(
    frame_rgb: np.ndarray,
    box:       Tuple[int, int, int, int],
    pad:       float = FACE_PAD,
    size:      int   = FRAME_SIZE,
) -> np.ndarray:
    """
    Crop the face region from a frame and resize to (size × size).

    A padding margin is added around the bounding box so that the model
    sees some context (hair, chin, neck) rather than a tight face crop.

    Parameters
    ----------
    frame_rgb : (H, W, 3) uint8
    box       : (x1, y1, x2, y2) in pixels
    pad       : fractional padding around the box (default 0.20 = 20%)
    size      : output spatial size (default 224)

    Returns
    -------
    np.ndarray of shape (size, size, 3) uint8
    """
    H, W    = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad)
    py = int(bh * pad)

    # Apply padding, clamp to frame boundaries
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(W, x2 + px)
    y2 = min(H, y2 + py)

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        # Degenerate box — return resized full frame as fallback
        crop = frame_rgb

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def fallback_crop(frame_rgb: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """
    Resize the full frame when no face is detected.

    A centre crop (60% of the frame) is taken first to reduce background,
    then resized to (size × size).
    """
    H, W = frame_rgb.shape[:2]
    cx, cy = W // 2, H // 2
    half_w = int(W * 0.30)
    half_h = int(H * 0.30)
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(W, cx + half_w)
    y2 = min(H, cy + half_h)
    crop = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def _laplacian_variance(frame_rgb: np.ndarray) -> float:
    """Sharpness metric. Higher = sharper. Pech-Pacheco et al. (2000)."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_area_ratio(
    box:        Tuple[int, int, int, int],
    frame_h:    int,
    frame_w:    int,
) -> float:
    """Fraction of frame area occupied by the face bounding box."""
    x1, y1, x2, y2 = box
    face_area  = (x2 - x1) * (y2 - y1)
    frame_area = frame_h * frame_w
    return face_area / frame_area if frame_area > 0 else 0.0


def _grade(value: float, good_thresh: float, degraded_thresh: float) -> str:
    """Apply a two-threshold grading scale."""
    if value >= good_thresh:
        return "good"
    elif value >= degraded_thresh:
        return "degraded"
    else:
        return "poor"


def _worst(grades: List[str]) -> str:
    """Return the worst grade from a list (poor > degraded > good)."""
    if "poor"     in grades: return "poor"
    if "degraded" in grades: return "degraded"
    return "good"


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def process_clip(
    frames:   np.ndarray,
    detector: FaceDetector,
    size:     int = FRAME_SIZE,
    pad:      float = FACE_PAD,
) -> Tuple[np.ndarray, SignalQuality, dict]:
    """
    Run face detection and cropping on all frames of a clip.

    Parameters
    ----------
    frames   : (T, H, W, 3) uint8 RGB
    detector : FaceDetector instance
    size     : output spatial size (default 224)
    pad      : padding fraction around face box (default 0.20)

    Returns
    -------
    cropped_frames : np.ndarray (T, size, size, 3) uint8
    quality        : "good" | "degraded" | "poor"
    metrics        : dict with raw metric values for logging/debugging
        {
          "face_detection_rate" : float,   # fraction of frames with a face
          "mean_laplacian_var"  : float,   # average sharpness across frames
          "mean_face_area_ratio": float,   # avg face area as fraction of frame
          "per_frame_detected"  : list,    # bool per frame
        }
    """
    T, H, W, _ = frames.shape
    cropped     = []
    detected     = []
    lap_vars     = []
    area_ratios  = []

    for frame in frames:
        boxes   = detector.detect(frame)
        lap_var = _laplacian_variance(frame)
        lap_vars.append(lap_var)

        if boxes:
            best_box = boxes[0]   # largest face
            area_ratios.append(_face_area_ratio(best_box, H, W))
            crop = crop_face(frame, best_box, pad=pad, size=size)
            detected.append(True)
        else:
            area_ratios.append(0.0)
            crop = fallback_crop(frame, size=size)
            detected.append(False)

        cropped.append(crop)

    cropped_arr        = np.stack(cropped)               # (T, size, size, 3)
    face_detection_rate = sum(detected) / T
    mean_lap_var        = float(np.mean(lap_vars))
    mean_area_ratio     = float(np.mean([a for a, d in zip(area_ratios, detected) if d])) \
                          if any(detected) else 0.0

    # Grade each metric independently, then take worst
    grade_detect = _grade(face_detection_rate, _DETECT_RATE_GOOD, _DETECT_RATE_DEGRADED)
    grade_lap    = _grade(mean_lap_var,        _LAP_VAR_GOOD,     _LAP_VAR_DEGRADED)
    grade_area   = _grade(mean_area_ratio,     _FACE_AREA_GOOD,   _FACE_AREA_DEGRADED)
    quality      = _worst([grade_detect, grade_lap, grade_area])

    metrics = {
        "face_detection_rate":  face_detection_rate,
        "mean_laplacian_var":   mean_lap_var,
        "mean_face_area_ratio": mean_area_ratio,
        "grade_detection_rate": grade_detect,
        "grade_laplacian":      grade_lap,
        "grade_face_area":      grade_area,
        "per_frame_detected":   detected,
        "backend":              detector.backend,
    }

    return cropped_arr, quality, metrics


# ---------------------------------------------------------------------------
# Assess quality only (no cropping) — used for dataset-level audit
# ---------------------------------------------------------------------------

def assess_clip_quality(
    frames:   np.ndarray,
    detector: FaceDetector,
) -> Tuple[SignalQuality, dict]:
    """
    Assess video signal quality without returning cropped frames.
    Useful for pre-filtering or dataset-level analysis.

    Parameters
    ----------
    frames   : (T, H, W, 3) uint8 RGB
    detector : FaceDetector instance

    Returns
    -------
    quality : "good" | "degraded" | "poor"
    metrics : dict (same keys as process_clip metrics)
    """
    _, quality, metrics = process_clip(frames, detector)
    return quality, metrics


# ---------------------------------------------------------------------------
# Quick smoke test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    print("=== Initialising FaceDetector ===")
    detector = FaceDetector()
    print(f"Backend: {detector.backend}")

    # Build a test clip: read a real video if a path is given, else use a
    # synthetic random clip (self-contained — no dataset module / manifest needed).
    if len(sys.argv) > 1:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(sys.argv[1])
        _frames = []
        while True:
            ok, f = cap.read()
            if not ok or f is None:
                break
            _frames.append(_cv2.cvtColor(f, _cv2.COLOR_BGR2RGB))
        cap.release()
        idx = np.linspace(0, len(_frames) - 1, 16, dtype=int)
        frames = np.stack([_frames[i] for i in idx])
        print(f"Loaded {len(_frames)} frames from {sys.argv[1]}, sampled 16")
    else:
        frames = (np.random.rand(16, 480, 640, 3) * 255).astype(np.uint8)
        print("No video path given — using a synthetic (16, 480, 640, 3) random clip")

    cropped, quality, metrics = process_clip(frames, detector)

    print(f"\n{'Quality':<10} {'Det.Rate':<10} {'LapVar':<10} {'AreaRatio':<12} {'Shape'}")
    print("-" * 60)
    print(
        f"{quality:<10} "
        f"{metrics['face_detection_rate']:<10.2f} "
        f"{metrics['mean_laplacian_var']:<10.1f} "
        f"{metrics['mean_face_area_ratio']:<12.3f} "
        f"{tuple(cropped.shape)}"
    )

    print("\n✓ Face detector smoke test complete")
