from pathlib import Path
import argparse
import numpy as np
import tifffile
from skimage.filters import gaussian
from skimage.util import view_as_windows

CROP_SIZE = 400
STEP = 200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing both images and masks."
    )
    parser.add_argument(
        "--mask-key",
        default="_mask.tif",
        help="Suffix used to identify mask files (default: _mask.tif)."
    )
    parser.add_argument(
        "--keep-empty-prob",
        type=float,
        default=0.3,
        help="Probability to keep empty-mask crops (default: 0.3)."
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=0.5,
        help="Gaussian blur sigma applied to images before cropping (default: 0.5)."
    )
    return parser.parse_args()


def _mask_base_name(path, mask_key):
    key_lower = mask_key.lower()
    name_lower = path.name.lower()
    if name_lower.endswith(key_lower):
        return path.name[:-len(mask_key)]
    stem_lower = path.stem.lower()
    if stem_lower.endswith(key_lower):
        return path.stem[:-len(mask_key)]
    return None


def process_and_save(data_dir, mask_key, keep_empty_prob, blur_sigma):
    data_dir = Path(data_dir)
    out_dir = data_dir / "crops_lean"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tiffs = sorted(list(data_dir.glob("*.tif")))
    mask_map = {}
    img_files = []
    for f in all_tiffs:
        base = _mask_base_name(f, mask_key)
        if base is not None:
            mask_map[base.lower()] = f
        else:
            img_files.append(f)

    print(f"Found {len(img_files)} source images.")

    for img_path in img_files:
        stem = img_path.stem
        match = mask_map.get(stem.lower())
        if not match:
            continue

        img = tifffile.imread(img_path)
        mask = tifffile.imread(match)

        if img.ndim == 3: img = img[0]
        if mask.ndim == 3: mask = mask[0]

        # Standardize
        img = img.astype(np.float32)
        img = np.floor_divide(img, 16)
        if blur_sigma and blur_sigma > 0:
            img = gaussian(img, sigma=blur_sigma, preserve_range=True).astype(np.float32, copy=False)

        # Pad
        h, w = img.shape
        pad_h = (CROP_SIZE - (h % CROP_SIZE)) % CROP_SIZE
        pad_w = (CROP_SIZE - (w % CROP_SIZE)) % CROP_SIZE

        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Crop
        img_patches = view_as_windows(img, (CROP_SIZE, CROP_SIZE), step=STEP)
        mask_patches = view_as_windows(mask, (CROP_SIZE, CROP_SIZE), step=STEP)

        img_patches = img_patches.reshape(-1, CROP_SIZE, CROP_SIZE)
        mask_patches = mask_patches.reshape(-1, CROP_SIZE, CROP_SIZE)

        count = 0
        for i, (ip, mp) in enumerate(zip(img_patches, mask_patches)):
            # Only save if there is data
            keep_empty = (mp.max() == 0) and (np.random.rand() < keep_empty_prob)
            if mp.max() > 0 or ip.max() > 5 or keep_empty:
                save_name = f"{img_path.stem}_{i:04d}.tif"
                tifffile.imwrite(out_dir / save_name, ip.astype(np.float32))
                tifffile.imwrite(out_dir / f"{img_path.stem}_{i:04d}{mask_key}", mp.astype(np.uint8))
                count += 1

        print(f"  {img_path.name}: Generated {count} crops.")


if __name__ == "__main__":
    args = parse_args()
    process_and_save(args.data_dir, args.mask_key, args.keep_empty_prob, args.blur_sigma)
