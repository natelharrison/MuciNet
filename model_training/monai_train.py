import os
import re
import argparse
from pathlib import Path
import torch
import wandb
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from monai.data import CacheDataset
from monai.losses import TverskyLoss
from monai.networks.nets import AttentionUnet
from monai.utils import set_determinism
from monai.transforms import (
    Compose, ToTensorD, RandRotateD, RandFlipD, Lambdad, MapTransform,
    EnsureTypeD, RandGaussianNoiseD, RandAdjustContrastD,
    RandScaleIntensityD, RandShiftIntensityD, RandAffined, NormalizeIntensityd
)
from monai.networks.nets import DynUNet


BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 2000
VAL_INTERVAL = 5
NUM_WORKERS = 4
PATIENCE = 128
MIN_DELTA = 1e-4


class LoadTiff(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = tifffile.imread(d[key])
            if arr.ndim == 2:
                arr = arr[None, ...]
            d[key] = arr.astype(np.float32)
        return d


def binarize_mask(x):
    return (x > 0).astype(np.float32)


def check_loader_outputs(loader, output_path="check_batch.png"):
    """Saves a plot of the first batch and uploads it to WandB."""
    batch = next(iter(loader))
    img = batch["image"][0, 0].cpu().numpy()
    mask = batch["mask"][0, 0].cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im0 = ax[0].imshow(img, cmap='gray')
    ax[0].set_title(f"Normalized Image\nMin: {img.min():.2f}, Max: {img.max():.2f}")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(mask, cmap='jet')
    ax[1].set_title("Mask (Binary)")
    plt.colorbar(im1, ax=ax[1])

    plt.savefig(output_path)
    wandb.log({"data_checks/sample_batch": wandb.Image(output_path)})
    print(f"Inspection image saved to {output_path} and uploaded to WandB.")
    plt.close()


def get_source_id_from_crop(path_str: str) -> str:
    stem = Path(path_str).stem
    m = re.match(r"^(.*)_\d{4}$", stem)
    return m.group(1) if m else stem


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, scheduler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch_data in loader:
        inputs = batch_data["image"].to(device, non_blocking=True)
        labels = batch_data["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
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
    parser.add_argument("--dir", required=True, help="Directory containing crops.")
    parser.add_argument("--mask-key", default="_mask.tif", help="Suffix for masks.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision('high')

    # Speed optimization for fixed-size inputs (e.g., 512x512 tiles)
    torch.backends.cudnn.benchmark = True

    set_determinism(seed=42)  # keep reproducibility; can reduce speed somewhat
    args = parse_args()

    run_dir = Path(args.dir).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project='mucin-segmentation',
        name='AttentionUNet_Tversky_ImageSplit_PerTileDice',
        config={
            "architecture": "AttentionUNet",
            "loss_function": "Tversky(alpha=0.7, beta=0.3)",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=500, T_mult=2)",
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": EPOCHS,
            "val_interval": VAL_INTERVAL,
            "patience": PATIENCE,
            "image_size": 512,
            "split_type": "Image-wise Grouped",
            "dataset_dir": str(args.dir),
            "normalization": "NormalizeIntensityd (Z-Score)",
            "bit_depth_handling": "Raw 16-bit to float32",
            "compile": not args.no_compile,
            "device": DEVICE,
            "num_workers": NUM_WORKERS,
            "persistent_workers": NUM_WORKERS > 0,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    )

    data_dir = Path(args.dir)
    all_tiffs = sorted(list(data_dir.glob("*.tif")))

    images = [str(f) for f in all_tiffs if args.mask_key not in f.name]
    files = []
    for img_path in images:
        mask_path = img_path.replace(".tif", args.mask_key)
        if os.path.exists(mask_path):
            files.append({"image": img_path, "mask": mask_path})

    if not files:
        raise ValueError("No valid image/mask pairs found.")


    groups = [get_source_id_from_crop(f["image"]) for f in files]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(files, groups=groups))

    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]

    train_sources = set(get_source_id_from_crop(f["image"]) for f in train_files)
    val_sources = set(get_source_id_from_crop(f["image"]) for f in val_files)
    assert train_sources.isdisjoint(val_sources), "Leakage detected: train/val source overlap"

    print(f"[split] train tiles: {len(train_files)} | val tiles: {len(val_files)}")
    print(f"[split] unique source images: train={len(set(get_source_id_from_crop(f['image']) for f in train_files))} "
          f"| val={len(set(get_source_id_from_crop(f['image']) for f in val_files))}")

    print("\n[VAL TILE SAMPLES]")
    for item in val_files[:20]:
        print(Path(item["image"]).name)

    # train_files, val_files = train_test_split(
    #     files,
    #     test_size=0.2,
    #     random_state=42,
    #     shuffle=True,
    # )

    train_transforms = Compose([
        LoadTiff(keys=["image", "mask"]),
        Lambdad(keys=["mask"], func=binarize_mask),
        NormalizeIntensityd(keys=["image"]),  # Z-score normalization
        RandFlipD(keys=["image", "mask"], prob=0.3, spatial_axis=0),
        RandFlipD(keys=["image", "mask"], prob=0.3, spatial_axis=1),
        RandRotateD(keys=["image", "mask"], range_x=0.2, prob=0.3),
        RandAffined(
            keys=["image", "mask"],
            prob=0.2,
            rotate_range=0.1,
            translate_range=(6, 6),
            padding_mode="reflect"
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
        NormalizeIntensityd(keys=["image"]),
        EnsureTypeD(keys=["image", "mask"]),
        ToTensorD(keys=["image", "mask"]),
    ])

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=NUM_WORKERS
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=NUM_WORKERS
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
    )

    check_loader_outputs(train_loader)

    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2)
    ).to(DEVICE)

    # model = DynUNet(
    #     spatial_dims=2,
    #     in_channels=1,
    #     out_channels=1,
    #     kernel_size=[3, 3, 3, 3, 3],
    #     strides=[1, 2, 2, 2, 2],
    #     upsample_kernel_size=[2, 2, 2, 2],
    #     filters=[32, 64, 128, 256, 512],
    #     norm_name="instance",
    #     act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    #     deep_supervision=False,
    #     res_block=True,
    # ).to(DEVICE)

    if not args.no_compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile unavailable, continuing without compile: {e}")

    loss_func = TverskyLoss(sigmoid=True, alpha=0.7, beta=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    scaler = torch.amp.GradScaler('cuda' if DEVICE == "cuda" else 'cpu')

    best_dice = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_func, DEVICE, scaler, scheduler)
        wandb.log({"train/loss": avg_loss, "epoch": epoch})

        if epoch % VAL_INTERVAL == 0:
            model.eval()
            tile_dice_vals = []
            tile_dice_nonempty = []

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(DEVICE, non_blocking=True)
                    labels = batch["mask"].to(DEVICE, non_blocking=True)

                    probs = torch.sigmoid(model(inputs))
                    outputs = (probs > 0.5).float()

                    # Per-tile Dice (B,1,H,W) -> (B,)
                    inter = (outputs * labels).sum(dim=(1, 2, 3))
                    union = outputs.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))

                    dice = torch.where(
                        union > 0,
                        (2.0 * inter) / (union + 1e-7),
                        torch.ones_like(union)  # both pred and GT empty => Dice = 1
                    )

                    tile_dice_vals.extend(dice.detach().cpu().numpy().tolist())

                    # GT non-empty tiles only
                    gt_nonempty = labels.sum(dim=(1, 2, 3)) > 0
                    if gt_nonempty.any():
                        tile_dice_nonempty.extend(
                            dice[gt_nonempty].detach().cpu().numpy().tolist()
                        )

            val_dice_mean = float(np.mean(tile_dice_vals)) if tile_dice_vals else 0.0
            val_dice_median = float(np.median(tile_dice_vals)) if tile_dice_vals else 0.0
            val_dice_nonempty = float(np.mean(tile_dice_nonempty)) if tile_dice_nonempty else 0.0

            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                f"Val Dice mean(tile): {val_dice_mean:.4f} | "
                f"median(tile): {val_dice_median:.4f} | "
                f"non-empty(tile): {val_dice_nonempty:.4f}"
            )

            wandb.log({
                "val/dice_mean_tile": val_dice_mean,
                "val/dice_median_tile": val_dice_median,
                "val/dice_nonempty_tile": val_dice_nonempty,
                # keep a legacy key for your existing dashboard plots / early stopping
                "val/dice": val_dice_mean,
                "epoch": epoch,
            })

            # Choose one metric for model selection and keep it consistent
            val_dice = val_dice_mean

            if val_dice > best_dice + MIN_DELTA:
                best_dice = val_dice
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "lean_best.pth")
                print("  *** New Best Saved! ***")
            elif (epoch - best_epoch) >= PATIENCE:
                print(f"Early stopping at epoch {epoch:03d}")
                break

    wandb.finish()


if __name__ == "__main__":
    main()