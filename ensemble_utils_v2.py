import cv2
import numpy as np

INVERT = False
MIN_AREA = 40

def _to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

def _normalize_and_center(img):
    """Resize keeping aspect ratio, center by mass like MNIST loader."""
    # bounding box of nonzero pixels
    ys, xs = np.where(img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((28,28), np.float32)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    cropped = img[y1:y2+1, x1:x2+1]

    # scale to fit 20x20 box
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad to 28x28
    pad_left = (28 - new_w)//2
    pad_top  = (28 - new_h)//2
    canvas = np.zeros((28,28), np.uint8)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    # center of mass shift
    ys, xs = np.where(canvas > 0)
    if len(xs) and len(ys):
        cx, cy = xs.mean(), ys.mean()
        shift_x, shift_y = int(round(14 - cx)), int(round(14 - cy))
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        canvas = cv2.warpAffine(canvas, M, (28,28), flags=cv2.INTER_NEAREST)

    # normalize to [0,1]
    return (canvas.astype(np.float32) / 255.0).reshape(28,28,1)

INVERT = False      
AUTO_INVERT = True    

def preprocess_image(image):
    """Binary image with digits as WHITE (255) on BLACK (0)."""
    gray = _to_gray(image)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # user-forced inversion
    if INVERT:
        bw = 255 - bw

    # auto-invert if most pixels are white (likely white background, black ink)
    if AUTO_INVERT:
        white_count = (bw == 255).sum()
        black_count = (bw == 0).sum()
        if white_count > black_count:
            bw = 255 - bw

    return bw

def _split_box_by_projection(roi, min_chunk=12, min_fg=35):
    """Robust horizontal splitter for a binary ROI (0/255)."""
    H, W = roi.shape[:2]
    fg = (roi == 255).astype(np.uint8)
    proj = fg.sum(axis=0).astype(np.float32)

    # smooth & normalize
    proj = cv2.GaussianBlur(proj.reshape(1, -1), (1, 9), 0).ravel()
    maxv = proj.max()
    if maxv <= 0:
        return [(0, W)]
    proj /= maxv

    # This threshold balances splitting '23' vs. merging '5'
    thr = 0.20
    valleys = np.where(proj < thr)[0]
    if valleys.size == 0:
        return [(0, W)]

    # group consecutive valley indices
    groups = np.split(valleys, np.where(np.diff(valleys) > 4)[0] + 1)
    cuts = [int(g.mean()) for g in groups if 3 < g.mean() < W - 3]

    # build tentative segments from cuts
    bounds = [0] + cuts + [W]
    chunks = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) < max(min_chunk, 8):
            continue
        # verify each chunk has enough foreground mass
        sub = fg[:, a:b]
        if int(sub.sum()) < min_fg:
            continue
        chunks.append((a, b))

    return chunks if len(chunks) >= 2 else [(0, W)]


def segment_and_extract_digits(image):
    """Final balanced segmenter."""
    bw = preprocess_image(image)

    # help reconnect tiny gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw_closed = cv2.medianBlur(bw_closed, 3)

    # find contours (with hierarchy for holes)
    contours, hierarchy = cv2.findContours(bw_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, c in enumerate(contours):
            # Skip if it's a child (inner hole)
            if hierarchy[i][3] != -1:
                continue

            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 100 or w < 8 or h < 15:
                continue  # too small to be a digit

            mask = np.zeros_like(bw_closed)
            cv2.drawContours(mask, [c], -1, 255, -1)
            white_pixels = cv2.countNonZero(mask[y:y+h, x:x+w])
            fill_ratio = white_pixels / float(area + 1e-6)

            # Skip if too hollow (angled '1', loopy '2')
            if fill_ratio < 0.15 or fill_ratio > 0.95:
                continue

            # merge nearby contours
            merged_contour = False
            for j, (xb, yb, wb, hb) in enumerate(boxes):
                if (abs(x - xb) < 15 and abs(y - yb) < 15 and
                    abs((x + w) - (xb + wb)) < 20 and abs((y + h) - (yb + hb)) < 20):
                    # merge as part of same digit
                    nx, ny = min(x, xb), min(y, yb)
                    max_x, max_y = max(x + w, xb + wb), max(y + h, yb + hb)
                    nw = max_x - nx
                    nh = max_y - ny
                    boxes[j] = (nx, ny, nw, nh)
                    merged_contour = True
                    break
            if merged_contour:
                continue

            boxes.append((x, y, w, h))

    # sanity: drop ghost boxes
    valid_boxes = []
    h_img, w_img = bw_closed.shape
    for (x, y, w, h) in boxes:
        if w * h < 200 or h < 20:
            continue
        if y > h_img - 40:  # too low near edge
            continue
        valid_boxes.append((x, y, w, h))
    boxes = valid_boxes

    if not boxes:
        return [], [], bw

    boxes.sort(key=lambda b: b[0])

    # merge broken fragments of ONE digit only
    merged = []
    if boxes:
        merged = [boxes[0]]
        for (x, y, w, h) in boxes[1:]:
            lx, ly, lw, lh = merged[-1]
            gap = x - (lx + lw)
            h_avg = (h + lh) / 2.0
            vert_overlap = min(ly + lh, y + h) - max(ly, y)
            area = w * h
            last_area = lw * lh
            small_fragment = area < 0.35 * last_area
            inter_w = max(0, min(lx + lw, x + w) - max(lx, x))
            inter_h = max(0, min(ly + lh, y + h) - max(ly, y))
            inter = inter_w * inter_h
            iou = inter / float(lw * lh + w * h - inter + 1e-6)
            vertical_misalignment = abs(y - ly) > 0.45 * h_avg

            # merge if: (a) almost touching + aligned OR (b) small fragment very close
            if (
                ((gap < max(10, 0.18 * h_avg) and vert_overlap > 0.55 * h_avg) or iou > 0.25)
                and not vertical_misalignment
            ) or (small_fragment and gap < 15):
                nx, ny = min(lx, x), min(ly, y)
                nw = max(lx + lw, x + w) - nx
                nh = max(ly + lh, y + h) - ny
                merged[-1] = (nx, ny, nw, nh)
            else:
                merged.append((x, y, w, h))

    # split boxes that still contain multiple digits
    final_boxes = []
    for (x, y, w, h) in merged:
        roi_c = bw_closed[y:y+h, x:x+w]
        aspect = w / float(h + 1e-6)

        # Check if box is wide enough to *potentially* be multiple digits
        need_split = (w > 40 or aspect > 1.5)

        if need_split:
            parts = _split_box_by_projection(roi_c, min_chunk=max(10, w // 10), min_fg=max(35, (w*h)//80))

            # This is a strict fallback splitter.
            # Only run if projection failed AND aspect is very wide (like '100')
            if len(parts) == 1 and aspect > 2.2:
                ys, xs = np.where(roi_c == 255)
                if xs.size >= 50:
                    xs = xs.reshape(-1, 1).astype(np.float32)
                    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.1)
                    # Fixed typo: cv2.KMEANS_PP_CENTERS
                    _, labels, centers = cv2.kmeans(xs, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
                    
                    if centers is not None and len(centers) == 2:
                        cut = int(np.mean(centers))
                        if 8 < cut < w - 8:
                            parts = [(0, cut), (cut, w)]
                    else:
                        parts = [(0, w)]

            for a, b in parts:
                ww = b - a
                if ww < max(10, w // 12):
                    continue
                # verify with foreground mass on the ORIGINAL bw
                sub = (bw[y:y+h, x+a:x+b] == 255).astype(np.uint8)
                if int(sub.sum()) < max(35, (h * ww) // 90):
                    continue
                final_boxes.append((x + a, y, ww, h))
        else:
            final_boxes.append((x, y, w, h))

    final_boxes.sort(key=lambda b: b[0])

    # build 28×28 and drop empty slices
    digits = []
    cleaned_boxes = []
    for (x, y, w, h) in final_boxes:
        roi = bw[y:y+h, x:x+w]  # use ORIGINAL bw
        
        # This check prevents tiny noise fragments from becoming '0's
        if np.count_nonzero(roi == 255) < 50:
            continue
            
        roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        norm = _normalize_and_center(roi)
        if norm.sum() < 1e-6:  # extra guard against all-zero tiles
            continue
        digits.append(norm)
        cleaned_boxes.append((x, y, w, h))

    print(f"[DEBUG] boxes={cleaned_boxes} (count={len(cleaned_boxes)})")
    return digits, cleaned_boxes, bw


def ensemble_predict(img28x28, cnn, dense, knn, weights=(0.5, 0.3, 0.2)):
    """Combines CNN, Dense, and KNN outputs."""
    try:
        # CNN prediction
        p_cnn = cnn.predict(img28x28.reshape(1, 28, 28, 1), verbose=0)[0]

        # Dense prediction
        flat = img28x28.reshape(1, -1)
        p_dense = dense.predict(flat, verbose=0)[0]

        # KNN prediction
        pred_knn = int(knn.predict(flat)[0])
        num_classes = max(len(p_cnn), len(p_dense))
        p_knn = np.zeros(num_classes)
        if 0 <= pred_knn < num_classes:
            p_knn[pred_knn] = 1.0

        # Align array sizes if different
        if len(p_dense) != len(p_cnn):
            diff = abs(len(p_dense) - len(p_cnn))
            if len(p_dense) < len(p_cnn):
                p_dense = np.pad(p_dense, (0, diff))
            else:
                p_cnn = np.pad(p_cnn, (0, diff))
                # Need to pad p_knn as well
                p_knn = np.pad(p_knn, (0, diff))

        # Weighted average (FIXED: re-added KNN)
        w_sum = sum(weights)
        weights = np.array(weights) / w_sum
        combined = (p_cnn * weights[0] + p_dense * weights[1] + p_knn * weights[2])
        pred_class = int(np.argmax(combined))

        # Debug print (FIXED: re-added KNN)
        print(f"[Ensemble DEBUG] CNN={np.argmax(p_cnn)}, Dense={np.argmax(p_dense)}, "
              f"KNN={pred_knn} → Final={pred_class}")

        return pred_class, combined

    except Exception as e:
        print(f"[Ensemble Error] {e}")
        return "?", None

