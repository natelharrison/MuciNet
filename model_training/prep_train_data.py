from pathlib import Path
import argparse
import numpy as np
import tifffile
from skimage.util import view_as_windows

CROP_SIZE = 512
STEP = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing images and masks.")
    parser.add_argument("--mask-key", default="_mask.tif", help="Suffix for mask files.")
    parser.add_argument("--keep-empty-prob", type=float, default=0.3, help="Prob to keep empty crops.")
    return parser.parse_args()


def _mask_base_name(path, mask_key):
    if path.name.endswith(mask_key):
        return path.name[:-len(mask_key)]
    return None


def compute_pad(length, crop_size, step):
    """
    Pad so that sliding windows of size `crop_size` with stride `step`
    exactly cover the padded dimension.

    We want:
        (padded_length - crop_size) % step == 0
    and padded_length >= crop_size
    """
    if length < crop_size:
        return crop_size - length

    rem = (length - crop_size) % step
    return 0 if rem == 0 else (step - rem)


def process_and_save(data_dir, mask_key, keep_empty_prob):
    data_dir = Path(data_dir)
    out_dir = data_dir / "crops_lean"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tiffs = sorted([f for f in data_dir.glob("*.tif") if out_dir not in f.parents])

    mask_map = {}
    img_files = []
    for f in all_tiffs:
        base = _mask_base_name(f, mask_key)
        if base is not None:
            mask_map[base] = f
        else:
            img_files.append(f)

    print(f"Found {len(img_files)} source images.")

    for img_path in img_files:
        match = mask_map.get(img_path.stem)
        if not match:
            continue

        img = tifffile.imread(img_path)
        mask = tifffile.imread(match)

        if img.ndim == 3:
            img = img[0]
        if mask.ndim == 3:
            mask = mask[0]

        img = img.astype(np.float32)

        h, w = img.shape
        pad_h = compute_pad(h, CROP_SIZE, STEP)
        pad_w = compute_pad(w, CROP_SIZE, STEP)

        if pad_h > 0 or pad_w > 0:
            # Image: reflect padding avoids artificial black borders
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
            # Mask: constant zero padding is correct for "outside image = background"
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        # Optional sanity check (can remove later)
        assert img.shape[0] >= CROP_SIZE and img.shape[1] >= CROP_SIZE
        assert (img.shape[0] - CROP_SIZE) % STEP == 0, f"Height not stride-aligned: {img.shape[0]}"
        assert (img.shape[1] - CROP_SIZE) % STEP == 0, f"Width not stride-aligned: {img.shape[1]}"

        img_patches = view_as_windows(img, (CROP_SIZE, CROP_SIZE), step=STEP)
        mask_patches = view_as_windows(mask, (CROP_SIZE, CROP_SIZE), step=STEP)

        img_patches = img_patches.reshape(-1, CROP_SIZE, CROP_SIZE)
        mask_patches = mask_patches.reshape(-1, CROP_SIZE, CROP_SIZE)

        count = 0
        for i, (ip, mp) in enumerate(zip(img_patches, mask_patches)):
            keep_empty = (mp.max() == 0) and (np.random.rand() < keep_empty_prob)
            if mp.max() > 0 or ip.max() > 80 or keep_empty:
                save_id = f"{img_path.stem}_{i:04d}"
                tifffile.imwrite(out_dir / f"{save_id}.tif", ip.astype(np.float32))
                tifffile.imwrite(out_dir / f"{save_id}{mask_key}", mp.astype(np.uint8))
                count += 1

        print(
            f"  {img_path.name}: Generated {count} crops "
            f"(orig={h}x{w}, pad=({pad_h},{pad_w}), padded={img.shape[0]}x{img.shape[1]})"
        )


if __name__ == "__main__":
    args = parse_args()
    process_and_save(args.data_dir, args.mask_key, args.keep_empty_prob)