import os
import argparse
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset
from monai.losses import TverskyLoss
from monai.networks.nets import AttentionUnet
from monai.utils import set_determinism
from monai.transforms import (
    Compose, ToTensorD, RandRotateD, RandFlipD, Lambdad, MapTransform,
    EnsureTypeD, RandGaussianNoiseD, RandAdjustContrastD,
    RandScaleIntensityD, RandShiftIntensityD, RandAffined
)

BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 1200
VAL_INTERVAL = 10
NUM_WORKERS = 8
PATIENCE = 128
MIN_DELTA = 1e-4


class LoadTiff(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = tifffile.imread(d[key])
            if arr.ndim == 2: arr = arr[None, ...]
            d[key] = arr.astype(np.float32)
        return d


def binarize_mask(x): return (x > 0).astype(np.float32)


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, scheduler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch_data in loader:
        # Non-blocking allows transfer while GPU computes
        inputs = batch_data["image"].to(device, non_blocking=True)
        labels = batch_data["mask"].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(inputs)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing images and masks; masks are identified by --mask-key."
    )
    parser.add_argument(
        "--mask-key",
        default="_mask.tif",
        help="Suffix used to identify mask files (default: _mask.tif)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (default: 16)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Learning rate (default: 2e-4)."
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for faster startup/debugging."
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


def main():
    torch.set_float32_matmul_precision('high')

    set_determinism(seed=42)
    args = parse_args()
    run_dir = Path(args.dir).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    batch_size = args.batch_size
    lr = args.lr

    data_dir = Path(args.dir)
    all_tiffs = sorted(list(data_dir.glob("*.tif")))
    mask_map = {}
    image_paths = []
    for f in all_tiffs:
        base = _mask_base_name(f, args.mask_key)
        if base is not None:
            mask_map[base.lower()] = f
        else:
            image_paths.append(str(f))
    mask_paths = []
    for ip in image_paths:
        stem = Path(ip).stem.lower()
        match = mask_map.get(stem)
        if match:
            mask_paths.append(str(match))
        else:
            mask_paths.append(None)
    pairs = [(im, mk) for im, mk in zip(image_paths, mask_paths) if mk]
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    if not image_paths: raise ValueError("No images found! Run Script 1 first.")

    files = [{"image": im, "mask": mk} for im, mk in zip(image_paths, mask_paths)]
    train_files, val_files = train_test_split(files, test_size=0.15, random_state=42)

    train_transforms = Compose([
        LoadTiff(keys=["image", "mask"]),
        Lambdad(keys=["mask"], func=binarize_mask),
        RandFlipD(keys=["image", "mask"], prob=0.3, spatial_axis=0),
        RandFlipD(keys=["image", "mask"], prob=0.3, spatial_axis=1),
        RandRotateD(keys=["image", "mask"], range_x=0.2, prob=0.3),
        RandAffined(
            keys=["image", "mask"],
            prob=0.2,
            rotate_range=0.1,
            translate_range=(6, 6),
            padding_mode="reflect",
        ),
        RandGaussianNoiseD(keys=["image"], prob=0.1, std=0.03),
        RandAdjustContrastD(keys=["image"], prob=0.3, gamma=(0.85, 1.25)),
        RandScaleIntensityD(keys=["image"], prob=0.2, factors=0.05),
        RandShiftIntensityD(keys=["image"], prob=0.2, offsets=0.03),
        EnsureTypeD(keys=["image", "mask"]),
        ToTensorD(keys=["image", "mask"]),
    ])

    val_transforms = Compose([
        LoadTiff(keys=["image", "mask"]),
        Lambdad(keys=["mask"], func=binarize_mask),
        EnsureTypeD(keys=["image", "mask"]),
        ToTensorD(keys=["image", "mask"]),
    ])

    # CacheDataset + num_workers=8 ensures GPU never waits for HDD
    print("Caching dataset...")
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)

    train_batch = min(batch_size, len(train_ds)) if len(train_ds) > 0 else batch_size
    val_batch = min(batch_size, len(val_ds)) if len(val_ds) > 0 else batch_size
    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, drop_last=False, persistent_workers=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, persistent_workers=True,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = AttentionUnet(spatial_dims=2, in_channels=1, out_channels=1, channels=(32, 64, 128, 256, 512),
                          strides=(2, 2, 2, 2)).to(DEVICE)

    if not args.no_compile:
        # *** COMPILE MODEL ***
        print("Compiling model for Ada architecture...")
        model = torch.compile(model)

    loss_func = TverskyLoss(sigmoid=True, alpha=0.7, beta=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')

    best_dice = -1.0
    best_epoch = 0
    print(f"Starting Training on {len(train_files)} crops (Batch {train_batch})...")

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_func, DEVICE, scaler, scheduler)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

        if epoch % VAL_INTERVAL == 0:
            model.eval()
            dice_vals = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
                    outputs = (torch.sigmoid(model(inputs)) > 0.5).float()
                    inter = (outputs * labels).sum()
                    union = outputs.sum() + labels.sum()
                    dice = (2. * inter) / (union + 1e-7) if union > 0 else 1.0
                    dice_vals.append(dice.item())

            val_dice = np.mean(dice_vals)
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f}")

            if val_dice > best_dice + MIN_DELTA:
                best_dice = val_dice
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "lean_best.pth")
                print(f"  *** New Best Saved! ***")
            elif (epoch - best_epoch) >= PATIENCE:
                print(f"Early stopping at epoch {epoch:03d} (no val dice improvement in {PATIENCE} checks).")
                break


if __name__ == "__main__":
    main()
