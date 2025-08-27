import os
import cv2
import numpy as np

BASE_DIR = Path(__file__).parent.absolute()
IN_DIR   = os.path.join(BASE_DIR, "preprocessed_task1")
OUT_DIR  = os.path.join(BASE_DIR, "task2_overlays")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Contours segmentation ---
def segment_contours(bw, dilate_kernel=(3,3), min_area=30):
    dil = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel), 1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area and h > 6 and w > 3:
            boxes.append((x,y,w,h))
    return sorted(boxes, key=lambda b: b[0])

# --- Splitter for wide blobs ---
def projection_split(sub_bw, min_run=6):
    v = (sub_bw.sum(axis=0) > 0).astype(np.uint8)
    runs, start = [], None
    for i, val in enumerate(v):
        if val and start is None:
            start = i
        if not val and start is not None:
            if i - start >= min_run:
                runs.append((start, i))
            start = None
    if start is not None and len(v)-start >= min_run:
        runs.append((start, len(v)))
    return runs

# --- Main segmentation ---
def segment_digits(img_gray):
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    boxes = segment_contours(bw)

    refined = []
    for (x,y,w,h) in boxes:
        if w > 1.5 * h:  # only split if clearly wide
            sub = bw[y:y+h, x:x+w]
            splits = projection_split(sub, min_run=6)
            if len(splits) >= 2:
                for sx, ex in splits:
                    refined.append((x+sx, y, ex-sx, h))
                continue
        refined.append((x,y,w,h))

    refined = sorted(refined, key=lambda b: b[0])

    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for (x,y,w,h) in refined:
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 1)  # 👈 thinner box

    return refined, vis

# --- Run on all Task 1 images ---
for fname in os.listdir(IN_DIR):
    if not fname.lower().endswith(".png"):
        continue
    path = os.path.join(IN_DIR, fname)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    boxes, vis = segment_digits(img_gray)
    base = os.path.splitext(fname)[0]

    out_path = os.path.join(OUT_DIR, f"{base}_boxes.png")
    cv2.imwrite(out_path, vis)

    print(f"[OK] {fname}: {len(boxes)} digits detected → {out_path}")

print("\nTask 2 complete. Overlays with thin boxes saved in:", OUT_DIR)
