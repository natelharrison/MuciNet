import tifffile
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from pathlib import Path
from tqdm import tqdm

MIN_DYNAMIC_RANGE = 30
MONTAGE_COLS = 4


def get_output_root(input_root: Path) -> Path:
    """returns the output root folder under the dataset root directory."""
    return input_root / "mucinet-results"


def imread(path):
    return np.squeeze(tifffile.imread(path))


def imwrite(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Help downstream TIFF readers (libtiff-based) interpret overlays correctly.
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] in (3, 4) and image.dtype == np.uint8:
        tifffile.imwrite(path, image, photometric="rgb")
    else:
        tifffile.imwrite(path, image)


def get_foreground(image, mask):
    """returns image with background pixels zeroed out."""
    return np.where(mask != 0, image, 0)


def get_background(image, mask):
    """returns image with foreground (cells/network) zeroed out."""
    return np.where(mask == 0, image, 0)


def create_overlay(raw, mask, color=(0, 255, 255)):
    """creates a cyan overlay of the mask onto the raw image."""
    if raw.dtype != np.uint8:
        raw = cv.normalize(raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    rgb = cv.cvtColor(raw, cv.COLOR_GRAY2RGB)
    rgb[mask > 0] = color
    return rgb


def _enhance_overlay_for_montage(image: np.ndarray) -> np.ndarray:
    """gentle contrast stretch and stronger cyan overlay without color shifts."""
    img = image.astype(np.float32, copy=False)

    if img.ndim == 2:
        p5, p95 = np.percentile(img, (5, 95))
        if p95 > p5:
            img = (img - p5) / (p95 - p5)
        img = np.clip(img, 0, 1) ** 0.9
        return (img * 255.0).astype(np.uint8)

    lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])
    p5, p95 = np.percentile(lum, (5, 95))
    if p95 > p5:
        lum = (lum - p5) / (p95 - p5)
    lum = np.clip(lum, 0, 1) ** 0.9
    base = (lum[..., None] * 255.0).astype(np.uint8)
    out = np.repeat(base, 3, axis=2)

    mask_pixels = (img[..., 1] > 150) & (img[..., 2] > 150) & (img[..., 0] < 100)
    if np.any(mask_pixels):
        alpha = 0.75
        cyan = np.array([0, 255, 255], dtype=np.uint8)
        out[mask_pixels] = (out[mask_pixels] * (1 - alpha) + cyan * alpha).astype(np.uint8)
    return out


def create_montage(
    trial_name,
    image_paths,
    output_path,
    group_by_genotype=True,
    enhance=False,
    groups=None,
    dpi=200,          # kept for compatibility; unused in canvas renderer
    downscale=0.5,
):
    """
    Create a montage as a single image canvas with GROUP HEADERS (folder names) only.
    No per-image titles. Uses OpenCV canvas rendering (robust, no matplotlib layout issues).
    """
    if not image_paths:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize group format / auto-group by parent folder name
    if groups is not None:
        groups = [
            (g.get("paths", []), g.get("color", "#555555"), g.get("label"), g.get("spacer_rows", 0))
            for g in groups
        ]
    else:
        grouped = {}
        for p in sorted(image_paths):
            p = Path(p)
            label = p.parent.name  # <-- folder name shown above each group
            grouped.setdefault(label, []).append(p)

        groups = []
        for label, paths in grouped.items():
            groups.append((paths, "#333333", label, 1))  # spacer row between groups

    def _hex_to_bgr(hex_color):
        if not isinstance(hex_color, str):
            return (80, 80, 80)
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return (80, 80, 80)
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (b, g, r)

    def _to_rgb_u8(img):
        """Display normalization to RGB uint8."""
        img = np.asarray(img)

        if enhance:
            img = _enhance_overlay_for_montage(img)

        # Simple min-max normalization for display
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        elif img.ndim == 3 and img.shape[-1] == 3:
            pass
        else:
            img = np.squeeze(img)
            if img.ndim == 2:
                img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"Unsupported image shape for montage: {img.shape}")

        if downscale and downscale < 1.0:
            h, w = img.shape[:2]
            new_w = max(1, int(w * downscale))
            new_h = max(1, int(h * downscale))
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

        return img

    # Find first valid image to determine tile size
    first_valid = None
    for group_paths, _, _, _ in groups:
        for p in group_paths:
            try:
                first_valid = _to_rgb_u8(imread(Path(p)))
                break
            except Exception:
                continue
        if first_valid is not None:
            break

    if first_valid is None:
        return

    tile_h, tile_w = first_valid.shape[:2]

    # Layout constants (pixels)
    margin_x = 30
    margin_y = 20
    top_title_h = 55
    cell_pad_x = 12
    cell_pad_y = 12
    group_header_h = 34
    spacer_h = 12  # visual gap between groups

    content_w = MONTAGE_COLS * (tile_w + cell_pad_x) - cell_pad_x

    # Build row plan
    row_specs = []
    for paths, color, label, spacer_rows in groups:
        paths = list(paths)
        if label:
            row_specs.append({"type": "group_header", "label": label, "color": color})
        if paths:
            for i in range(0, len(paths), MONTAGE_COLS):
                row_specs.append({"type": "tiles", "paths": paths[i:i + MONTAGE_COLS], "color": color})
        for _ in range(spacer_rows):
            row_specs.append({"type": "spacer"})

    if not row_specs:
        return

    # Compute canvas size
    total_h = margin_y + top_title_h + margin_y
    for row in row_specs:
        if row["type"] == "group_header":
            total_h += group_header_h
        elif row["type"] == "tiles":
            total_h += tile_h + cell_pad_y
        elif row["type"] == "spacer":
            total_h += spacer_h
    total_h += margin_y

    total_w = margin_x + content_w + margin_x
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # Trial title (top)
    cv.putText(
        canvas,
        str(trial_name),
        (margin_x, margin_y + 35),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv.LINE_AA,
    )

    y = margin_y + top_title_h + margin_y
    x0 = margin_x

    for row in row_specs:
        if row["type"] == "group_header":
            color_bgr = _hex_to_bgr(row["color"])
            label = str(row["label"])

            # Optional light divider line
            cv.line(canvas, (margin_x, y + 4), (total_w - margin_x, y + 4), (225, 225, 225), 1)

            cv.putText(
                canvas,
                label,
                (margin_x, y + 26),
                cv.FONT_HERSHEY_SIMPLEX,
                0.85,      # bigger so clearly visible
                color_bgr,
                2,
                cv.LINE_AA,
            )
            y += group_header_h
            continue

        if row["type"] == "spacer":
            y += spacer_h
            continue

        # tiles row (NO per-image titles)
        paths = row["paths"]

        for col_idx in range(MONTAGE_COLS):
            if col_idx >= len(paths):
                continue

            x = x0 + col_idx * (tile_w + cell_pad_x)
            p = Path(paths[col_idx])

            try:
                img = _to_rgb_u8(imread(p))
            except Exception:
                img = np.full((tile_h, tile_w, 3), 240, dtype=np.uint8)
                cv.putText(img, "error", (10, tile_h // 2),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)

            if img.shape[:2] != (tile_h, tile_w):
                img = cv.resize(img, (tile_w, tile_h), interpolation=cv.INTER_AREA)

            y0 = y
            y1 = y0 + tile_h
            x1 = x + tile_w
            canvas[y0:y1, x:x1] = img

            # thin border
            cv.rectangle(canvas, (x, y0), (x1 - 1, y1 - 1), (220, 220, 220), 1)

        y += tile_h + cell_pad_y

    imwrite(output_path, canvas)


def _worker(args):
    paths, func = args
    img_path, mask_path, input_root, output_root = paths

    image = imread(img_path)
    if mask_path is not None and mask_path.exists():
        mask = imread(mask_path)
    else:
        mask = np.ones(image.shape, dtype=np.uint8)

    results = func(image, mask)
    if not results:
        return

    owner_dir = img_path.parent
    try:
        rel_owner = owner_dir.relative_to(input_root)
    except ValueError:
        rel_owner = Path(owner_dir.name)
    results_root = output_root / rel_owner

    for key, data in results.items():
        if hasattr(data, "to_csv"):
            save_dir = results_root / "metrics"
            save_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(save_dir / f"{img_path.stem}_{key}", index=False)
        else:
            save_path = results_root / key / img_path.name
            imwrite(save_path, data)


def build_tasks(parent_dir, output_root=None):
    if output_root is None:
        output_root = get_output_root(parent_dir)
    tasks = []

    # Only consider TIFFs located directly in phenotype folders (ROOT/<TRIAL>/<PHENOTYPE>/*.tif).
    tiff_imgs = list(parent_dir.rglob("*.tif")) + list(parent_dir.rglob("*.tiff"))
    for img in sorted(set(tiff_imgs)):
        if "mucinet-results" in img.parts:
            continue
        try:
            rel = img.relative_to(parent_dir)
        except ValueError:
            continue
        # Require TIFF to be directly inside the phenotype folder.
        if len(rel.parts) != 3:
            continue
        tasks.append((img, None, parent_dir, output_root))
    return tasks


def run_pipeline(tasks, process_func):
    if not tasks:
        return
    print(f"processing {len(tasks)} files on a single process...")
    for t in tqdm(tasks, total=len(tasks)):
        _worker((t, process_func))