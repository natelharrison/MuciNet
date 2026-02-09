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
    """returns the output root folder outside the input data directory."""
    return input_root.parent / "output" / input_root.name


def imread(path):
    return np.squeeze(tifffile.imread(path))


def imwrite(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    dpi=200,
    downscale=0.5,
):
    """generates a grid montage for a trial."""
    if not image_paths: return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if groups is not None:
        groups = [
            (g.get("paths", []), g.get("color", "#555555"), g.get("label"), g.get("spacer_rows", 0))
            for g in groups
        ]
    elif group_by_genotype:
        wts = sorted([p for p in image_paths if 'wt' in get_genotype_from_name(p.name).lower()])
        muts = sorted([p for p in image_paths if 'mutant' in get_genotype_from_name(p.name).lower()])
        groups = [(wts, '#555555', None, 0), (muts, '#E63946', None, 0)]
    else:
        groups = [(sorted(image_paths), '#555555', None, 0)]

    total_rows = sum(
        math.ceil(len(group) / MONTAGE_COLS) + spacer
        for group, _, _, spacer in groups
        if group or spacer
    )

    if total_rows == 0: return

    fig, axes = plt.subplots(total_rows, MONTAGE_COLS, figsize=(MONTAGE_COLS * 3, total_rows * 3.5))
    fig.suptitle(f"{trial_name}", fontsize=20, fontweight='bold')

    axes_flat = axes.flatten() if total_rows * MONTAGE_COLS > 1 else [axes]
    current_idx = 0

    def _short_label(stem):
        clean = stem.replace("MAX_chan1_", "").replace("_", " ").strip()
        parts = clean.split()
        if len(parts) >= 2:
            return " ".join(parts[-2:])
        return clean

    label_specs = []

    def plot_group(paths, label_color, group_label, spacer_rows):
        nonlocal current_idx
        group_start_idx = current_idx
        for p in paths:
            ax = axes_flat[current_idx]
            try:
                img = imread(p)
                if enhance:
                    img = _enhance_overlay_for_montage(img)
                if downscale and downscale < 1.0:
                    h, w = img.shape[:2]
                    img = cv.resize(img, (int(w * downscale), int(h * downscale)), interpolation=cv.INTER_AREA)
                ax.imshow(img)
                ax.set_title(_short_label(p.stem), color=label_color, fontsize=8, fontweight='bold')
            except Exception:
                ax.text(0.5, 0.5, "error", ha='center')
            ax.axis('off')
            current_idx += 1

        while current_idx % MONTAGE_COLS != 0:
            axes_flat[current_idx].axis('off')
            current_idx += 1
        if spacer_rows:
            for _ in range(spacer_rows * MONTAGE_COLS):
                axes_flat[current_idx].axis('off')
                current_idx += 1
        if group_label:
            label_specs.append((axes_flat[group_start_idx], group_label, label_color))

    for group, color, label, spacer in groups:
        if group or spacer:
            plot_group(group, color, label, spacer)

    if label_specs:
        for ax, label, color in label_specs:
            pos = ax.get_position()
            y_center = (pos.y0 + pos.y1) / 2
            fig.text(0.01, y_center, label, ha='left', va='center', fontsize=9, fontweight='bold', color=color)
        plt.tight_layout(rect=(0.12, 0, 1, 0.95))
        plt.subplots_adjust(hspace=0.25)
    else:
        plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def get_genotype_from_name(filename):
    clean = filename.lower().replace('_', ' ').replace('-', ' ')
    if any(x in clean.split() for x in ['v', 'v1', 'v2']):
        return 'Mutant'
    return 'WT'


def _worker(args):
    paths, func = args
    img_path, mask_path, input_root, output_root = paths

    image = imread(img_path)
    if mask_path is not None and mask_path.exists():
        mask = imread(mask_path)
    else:
        mask = np.ones(image.shape, dtype=np.uint8)

    results = func(image, mask)
    if not results: return

    results_root = None
    for parent in img_path.parents:
        if parent.name == "MIPs":
            mips_owner = parent.parent
            try:
                rel_owner = mips_owner.relative_to(input_root)
            except ValueError:
                rel_owner = Path(mips_owner.name)
            results_root = output_root / rel_owner / "analysis_results"
            break
    if results_root is None:
        return

    for key, data in results.items():
        if hasattr(data, 'to_csv'):
            save_dir = results_root / 'quantification'
            save_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(save_dir / f"{img_path.stem}_{key}", index=False)
        else:
            save_path = results_root / key / img_path.name
            imwrite(save_path, data)


def build_tasks(parent_dir, output_root=None):
    if output_root is None:
        output_root = get_output_root(parent_dir)
    tasks = []
    for img in parent_dir.rglob("MIPs/chan1/*.tif"):
        tasks.append((img, None, parent_dir, output_root))
    return tasks


def run_pipeline(tasks, process_func):
    if not tasks: return
    print(f"processing {len(tasks)} files on a single process...")
    for t in tqdm(tasks, total=len(tasks)):
        _worker((t, process_func))
