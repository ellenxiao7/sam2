#!/usr/bin/env python3
"""
Batch Gauge Reader with AWS Rekognition
--------------------------------------
For every gauge image:
  1. Detect circular dial region via SAM.
  2. Crop around dial and detect needle.
  3. Call AWS Rekognition 'detect_text' on the crop.
  4. Save annotated output for visual QA and print angles/readings.
"""
from typing import List, Tuple
import os
import io
import glob
import json
import cv2
import boto3
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
from pathlib import Path
import csv
from datetime import datetime


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
IMAGE_DIR_BASE = "/mnt/projects-np/gauge-reading-poc/"
# OUTPUT_DIR = "/mnt/projects-np/gauge-reading-poc/outputs"
OUTPUT_DIR = "./outputs/test_images"
AWS_REGION = "ap-southeast-2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "batch_results_test_images.csv")


# ---------------------------------------------------------------------
# SAM SETUP
# ---------------------------------------------------------------------
sam2 = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# ---------------------------------------------------------------------
# AWS Rekognition client
# ---------------------------------------------------------------------
rekog = boto3.client("rekognition", region_name=AWS_REGION)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_sam_generator(type: str, image: np.ndarray):
    if type == "dial":

        H, W = image.shape[:2]
        img_area = H * W
        min_area_coarse = int(max(0.1 * img_area, 1500))

        return SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=10,
            points_per_batch=8,
            pred_iou_thresh=0.82,
            stability_score_thresh=0.94,
            stability_score_offset=0.70,
            box_nms_thresh=0.50,
            min_mask_region_area=min_area_coarse,
            use_m2m=True,
        )
    
    elif type == "needle":

        return SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=20,  # 48–64 keeps granularity down
            points_per_batch=16,
            pred_iou_thresh=0.80,  # be stricter
            stability_score_thresh=0.92,  # be stricter
            stability_score_offset=0.70,
            crop_n_layers=1,  # fewer scales → fewer tiny fragments
            crop_n_points_downscale_factor=2,
            box_nms_thresh=0.55,
            min_mask_region_area=600,  # absolute floor; we’ll also apply a face-relative floor next
            use_m2m=True,
        )
    else:
        raise ValueError(f"Unknown mask generator type: {type}")

def detect_text_with_rekognition(crop_rgb):
    """Call AWS Rekognition detect_text on a crop (numpy RGB image)."""
    _, buf = cv2.imencode(".png", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    response = rekog.detect_text(Image={'Bytes': buf.tobytes()})
    return response

def angle_from_center(px, py, cx, cy):
    dx, dy = px - cx, py - cy
    return (np.degrees(np.arctan2(dx, -dy)) + 360) % 360

CSV_FIELDS = [
    "timestamp",
    "status",                 # "ok" | "failed"
    "path",
    "error",
    # circle / crop
    "circle_cx", "circle_cy", "circle_r",
    "crop_x0", "crop_y0", "crop_x1", "crop_y1",
    "crop_cx", "crop_cy",     # center in crop coords
    # needle
    "needle_tip_x", "needle_tip_y",
    "needle_angle_deg",
    "needle_unwrapped_deg",
    # rekognition / labels / fit
    "rekog_text_count",
    "labels_used",            # e.g. "0|10|20|...|100"
    "label_angles_unwrapped", # e.g. "12.1|45.0|..."
    "fit_intercept", "fit_slope",
    # reading
    "reading_raw",
    "reading_clamped",
    # debug
    "notes",
]

def _empty_row():
    return {k: "" for k in CSV_FIELDS}

def _ensure_csv_header():
    if not os.path.exists(RESULTS_CSV):
        os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()

def _append_csv_row(row: dict):
    _ensure_csv_header()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writerow(row)


# ---------------------------------------------------------------------
# Rekognition text detection results parsing
# ---------------------------------------------------------------------
def parse_rekognition_text(crop,  angle_deg, rekog_resp, cx_crop, cy_crop, r, conf_thresh=90.0,):
    valid_tokens = {str(v) for v in range(0, 101, 10)}  # "0","10",...,"100"
    h, w = crop.shape[:2]

    def bbox_center_px(bb):
        # Rekognition bbox is normalized
        cx = (bb["Left"] + bb["Width"] / 2.0) * w
        cy = (bb["Top"] + bb["Height"] / 2.0) * h
        return cx, cy


    def label_angle_deg(cx_lbl, cy_lbl):
        dx = cx_lbl - cx_crop
        dy = cy_lbl - cy_crop
        return (np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0  # north=0°, CW+
    # 1) collect label -> angle
    vals, angs = [], []
    for td in rekog_resp["TextDetections"]:
        if td.get("Type") != "WORD":
            continue
        txt = td.get("DetectedText", "").strip()
        if txt not in valid_tokens:
            continue
        conf = float(td.get("Confidence", 0))
        if conf < conf_thresh:
            continue
        bb = td["Geometry"]["BoundingBox"]
        lx, ly = bbox_center_px(bb)
        ang = label_angle_deg(lx, ly)
        vals.append(int(txt))
        angs.append(ang)

    assert len(vals) >= 3, f"Need at least 3 tick labels; got {len(vals)}."

    # 2) sort by numeric value
    vals = np.array(vals)
    angs = np.array(angs)
    order = np.argsort(vals)
    vals = vals[order]
    angs = angs[order]

    # 3) unwrap angles to be monotonic with increasing values
    #    (we allow adding 360° to later items to avoid wrap-around)
    unwrap = angs.copy()
    for i in range(1, len(unwrap)):
        while unwrap[i] < unwrap[i - 1] - 180:
            unwrap[i] += 360
        while unwrap[i] > unwrap[i - 1] + 180:
            unwrap[i] -= 360

    # Now make fully increasing by lifting later angles if needed
    for i in range(1, len(unwrap)):
        if unwrap[i] <= unwrap[i - 1]:
            # push it just above previous (add k*360)
            k = int(np.floor((unwrap[i - 1] - unwrap[i]) / 360.0)) + 1
            unwrap[i] += 360.0 * k

    # 4) linear fit: reading ≈ a + b * angle_unwrapped
    #    (independent var = angle, dependent = value)
    A = np.vstack([np.ones_like(unwrap), unwrap]).T
    coeff, _, _, _ = np.linalg.lstsq(A, vals.astype(float), rcond=None)
    a, b = coeff  # reading = a + b * angle

    # 5) map needle angle -> unwrap it near the calibration span
    needle = float(angle_deg)

    # choose the unwrap for the needle that keeps it close to the fitted span
    span_lo, span_hi = unwrap.min(), unwrap.max()
    # try needle shifted by k*360 for k in {-1,0,1,2} and pick closest to span
    candidates = [needle + 360 * k for k in (-1, 0, 1, 2)]
    needle_unwrapped = min(
        candidates, key=lambda x: min(abs(x - span_lo), abs(x - span_hi))
    )

    reading = a + b * needle_unwrapped

    # clamp to labeled range (optionally extend a little if you want)
    reading_clamped = float(np.clip(reading, vals.min(), vals.max()))

    print(f"Labels used: {list(zip(vals.tolist(), np.round(unwrap, 1).tolist()))}")
    print(f"Fit: reading ≈ {a:.3f} + {b:.3f} * angle")
    print(
        f"Needle angle: {angle_deg:.1f}°  ->  reading ≈ {reading:.2f}  (clamped: {reading_clamped:.2f})"
    )
    return reading_clamped, dict(fit_intercept=a, fit_slope=b,
                                 vals=vals.tolist(),
                                 unwrap=np.round(unwrap,1).tolist(),
                                 needle_angle=angle_deg,
                                 needle_unwrapped=needle_unwrapped)

    
# ---------------------------------------------------------------------
# Circle detection
# ---------------------------------------------------------------------
def get_circle_prior(image, masks):
    H, W = image.shape[:2]
    rel_area_min, rel_area_max = 0.03, 0.60
    circ_min = 0.62
    border_margin = max(6, int(0.01 * min(H, W)))
    pts_all = []
    for m in masks:
        seg = (m['segmentation'].astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        per  = cv2.arcLength(cnt, True) + 1e-6
        circ = (4*np.pi*area)/(per*per)
        x,y,w,h = cv2.boundingRect(cnt)
        touches_border = (x<=border_margin or y<=border_margin or
                          x+w>=W-border_margin or y+h>=H-border_margin)
        if touches_border: continue
        rel_area = area / (H*W)
        if not (rel_area_min <= rel_area <= rel_area_max): continue
        if circ < circ_min: continue
        pts_all.append(cnt.reshape(-1,2))
    if not pts_all:
        raise RuntimeError("No circular mask found.")
    pts = np.vstack(pts_all).astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return float(cx), float(cy), float(r)

# ---------------------------------------------------------------------
# Needle detection
# ---------------------------------------------------------------------
def detect_needle(masks, cx, cy, r):
    best, best_score = None, -np.inf
    for m in masks:
        seg = (m['segmentation'].astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 80: continue
        dists = np.hypot(cnt[:,0,0]-cx, cnt[:,0,1]-cy)
        min_d, max_d = dists.min(), dists.max()
        passes_center = min_d < 0.10 * r
        if not passes_center: continue
        x,y,w,h = cv2.boundingRect(cnt)
        aspect = max(w,h)/(min(w,h)+1e-6)
        score = (aspect*0.5)+(max_d/r)+1.0
        if score > best_score:
            best_score, best = score, cnt
    if best is None:
        raise RuntimeError("Needle not found.")
    return best

# ---------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------
def process_image(img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing {name} ...")

    row = _empty_row()
    row["timestamp"] = datetime.utcnow().isoformat()
    row["path"] = str(img_path)
    row["status"] = "failed"  # default; set to "ok" on success

    try:
        # --- Load image ---
        image = np.array(Image.open(img_path).convert("RGB"))

        # --- Find dial circle ---
        mask_generator_dial = get_sam_generator("dial", image)
        masks = mask_generator_dial.generate(image)
        cx, cy, r = get_circle_prior(image, masks)

        # record circle
        row["circle_cx"] = f"{cx:.3f}"
        row["circle_cy"] = f"{cy:.3f}"
        row["circle_r"]  = f"{r:.3f}"

        # --- Crop around dial ---
        marg = int(0.05 * r)
        x0, y0 = max(0, int(cx - r - marg)), max(0, int(cy - r - marg))
        x1, y1 = min(image.shape[1], int(cx + r + marg)), min(image.shape[0], int(cy + r + marg))
        crop = image[y0:y1, x0:x1]
        cx_crop, cy_crop = cx - x0, cy - y0

        # record crop box
        row["crop_x0"] = str(x0)
        row["crop_y0"] = str(y0)
        row["crop_x1"] = str(x1)
        row["crop_y1"] = str(y1)
        row["crop_cx"] = f"{cx_crop:.3f}"
        row["crop_cy"] = f"{cy_crop:.3f}"

        # --- Detect needle ---
        mask_generator_needle = get_sam_generator("needle", crop)
        masks_in = mask_generator_needle.generate(crop)
        needle = detect_needle(masks_in, cx_crop, cy_crop, r)
        pts = needle.reshape(-1, 2)
        d = np.hypot(pts[:, 0] - cx_crop, pts[:, 1] - cy_crop)
        tip = pts[np.argmax(d)]
        angle_deg = (np.degrees(np.arctan2(tip[0] - cx_crop, -(tip[1] - cy_crop))) + 360) % 360

        # record needle
        row["needle_tip_x"] = f"{float(tip[0]):.3f}"
        row["needle_tip_y"] = f"{float(tip[1]):.3f}"
        row["needle_angle_deg"] = f"{angle_deg:.6f}"

        # --- AWS Rekognition detect_text ---
        rekog_resp = detect_text_with_rekognition(crop)
        with open(f"{OUTPUT_DIR}/{name}_rekog.json", "w") as f:
            json.dump(rekog_resp, f, indent=2)
        det_count = len(rekog_resp.get('TextDetections', []))
        print(f"  Rekognition found {det_count} text entries")
        row["rekog_text_count"] = str(det_count)

        # --- Parse Rekognition results to get reading ---
        reading_clamped = None
        reading_info = None
        try:
            # FIX: unpack tuple (reading, info)
            reading_clamped, reading_info = parse_rekognition_text(
                crop, angle_deg, rekog_resp, cx_crop, cy_crop, r, conf_thresh=90.0,
            )

            # record labels / fit / needle unwrap
            if reading_info:
                vals = reading_info.get("vals", [])
                unwrap = reading_info.get("unwrap", [])
                row["labels_used"] = "|".join(map(str, vals)) if vals else ""
                row["label_angles_unwrapped"] = "|".join(map(str, unwrap)) if unwrap else ""
                row["fit_intercept"] = f"{reading_info.get('fit_intercept', ''):.6f}" if "fit_intercept" in reading_info else ""
                row["fit_slope"] = f"{reading_info.get('fit_slope', ''):.6f}" if "fit_slope" in reading_info else ""
                if "needle_unwrapped" in reading_info:
                    row["needle_unwrapped_deg"] = f"{reading_info['needle_unwrapped']:.6f}"

        except Exception as e:
            print(f"  WARNING: Could not parse Rekognition results: {e}")
            row["notes"] = "parse_rekognition_text failed"
            row["error"] = str(e)

        # --- Visualization ---
        vis = crop.copy()
        cv2.drawContours(vis, [needle], -1, (255, 0, 0), 2)
        cv2.drawMarker(vis, (int(cx_crop), int(cy_crop)), (0, 255, 0),
                       cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.line(vis, (int(cx_crop), int(cy_crop)), (int(tip[0]), int(tip[1])), (255, 255, 0), 2)

        # "reading_clamped" is a float if parse succeeded
        if reading_clamped is not None:
            row["reading_raw"] = row["reading_clamped"] = f"{float(reading_clamped):.6f}"
            cv2.putText(vis, f"{float(reading_clamped):.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        plt.imsave(f"{OUTPUT_DIR}/{name}_annotated.png", vis)
        print(f"  Saved {name}_annotated.png\n")

        row["status"] = "ok"
        return row

    except Exception as e:
        # Any failure ends up here; we still write a CSV row.
        print(f"  ERROR while processing {name}: {e}")
        row["error"] = str(e)
        return row

# ---------------------------------------------------------------------
# Utility: collect images
# ---------------------------------------------------------------------
def collect_images_one_per_leaf(base_dir: str,
                                exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
                                ) -> List[Path]:
    """
    Return a list with exactly one image Path from each leaf directory under base_dir.
    A 'leaf' is a directory that has no subdirectories.
    For determinism, we pick the lexicographically first matching image in that leaf.
    """
    base = Path(base_dir)
    picks: List[Path] = []

    # Walk the tree; a leaf has no subdirectories
    for dirpath, dirnames, filenames in os.walk(base):
        if dirnames:
            continue  # not a leaf, keep descending

        # filter images (case-insensitive)
        imgs = [
            Path(dirpath) / fn
            for fn in filenames
            if Path(fn).suffix.lower() in exts
        ]
        if not imgs:
            continue

        # deterministic pick: first in sorted order
        imgs_sorted = sorted(imgs, key=lambda p: p.name.lower())
        picks.append(imgs_sorted[0])

    return picks

def collect_all_images(base_dir: str,
                       exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
                       ) -> List[Path]:
    """
    Return a list of ALL image Paths under base_dir (recursive).
    Matches extensions case-insensitively.
    """
    base = Path(base_dir)
    imgs: List[Path] = []

    for dirpath, _, filenames in os.walk(base):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                imgs.append(Path(dirpath) / fn)

    return imgs


# ---------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------
def main():
    base = IMAGE_DIR_BASE
    direc = f"{base}/test_images"
    # imgs = collect_images_one_per_leaf(base)
    imgs = collect_all_images(direc)
    print(f"Found {len(imgs)} images to process.")

    for p in imgs:
        row = None
        try:
            row = process_image(p)
        except Exception as e:
            # Fallback row if process_image threw before returning
            row = _empty_row()
            row["timestamp"] = datetime.now().isoformat()
            row["path"] = str(p)
            row["status"] = "failed"
            row["error"] = f"process_image() raised: {e}"

        _append_csv_row(row)

    print(f"Results written to: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
