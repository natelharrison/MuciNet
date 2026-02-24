import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext
from monai.networks.nets import AttentionUnet
from monai.inferers import SlidingWindowInferer
from skimage import morphology
import pipeline_utils
from monai.networks.nets import DynUNet


_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "model_training" / "lean_best.pth"
)
MODEL_PATH = _DEFAULT_MODEL_PATH if _DEFAULT_MODEL_PATH.exists() else None

ROI_SIZE = (512, 512)

PROB_THRESHOLD = 0.5
MIN_OBJECT_SIZE = 64
DEVICE = "auto"
SW_BATCH_SIZE = 16
SW_OVERLAP = 0.5


def normalize_to_8bit(image):
    """Robust 8-bit conversion for overlays."""
    if image.dtype == np.uint8:
        return image

    min_val, max_val = float(image.min()), float(image.max())
    if (max_val - min_val) < pipeline_utils.MIN_DYNAMIC_RANGE:
        return np.zeros_like(image, dtype=np.uint8)

    scaled = (image - min_val) / (max_val - min_val) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Match training preprocessing:
    - float32 cast
    - per-image normalization similar to MONAI NormalizeIntensityd
      (nonzero=False by default -> normalize all pixels)
    """
    im = image.astype(np.float32, copy=False)

    mean = float(im.mean())
    std = float(im.std())

    if std < 1e-8:
        return np.zeros_like(im, dtype=np.float32)

    im = (im - mean) / std
    return im.astype(np.float32, copy=False)


_MODEL = None
_INFERER = None
_DEVICE = None


def _get_device():
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def _load_model():
    global _MODEL, _INFERER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _INFERER, _DEVICE

    if MODEL_PATH is None or not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Provide --model-path to a valid .pth file."
        )

    _DEVICE = _get_device()

    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(_DEVICE)

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
    # ).to(_DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=_DEVICE)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[len("_orig_mod."):]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    _MODEL = model
    _INFERER = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        overlap=SW_OVERLAP,
        mode="gaussian",
    )
    return _MODEL, _INFERER, _DEVICE


def analyze_networks(image, mask):
    model, inferer, device = _load_model()
    im_norm = preprocess(image)

    inp = torch.from_numpy(im_norm[None, None, ...]).to(device=device, dtype=torch.float32)

    autocast_ctx = torch.amp.autocast("cuda") if device == "cuda" else nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            logits = inferer(inp, model)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    pred_mask = prob > PROB_THRESHOLD
    pred_mask = morphology.remove_small_objects(pred_mask, min_size=MIN_OBJECT_SIZE)
    mask_uint8 = (pred_mask.astype(np.uint8) * 255)

    raw_pixel_count = int(np.count_nonzero(pred_mask))

    stats = pd.DataFrame({
        "monai_mask_pixels": [raw_pixel_count],
        "model_path": [str(MODEL_PATH)],
        "roi_size": [tuple(ROI_SIZE)],
        "sw_batch_size": [SW_BATCH_SIZE],
        "sw_overlap": [SW_OVERLAP],
        "prob_threshold": [PROB_THRESHOLD],
        "min_object_size": [MIN_OBJECT_SIZE],
        "device": [device],
        "preprocess": ["zscore_per_image"],
    })

    return {
        "network_binary": mask_uint8,
        "network_overlay": pipeline_utils.create_overlay(normalize_to_8bit(image), mask_uint8),
        "network_stats.csv": stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MONAI network segmentation on input TIFFs."
    )
    parser.add_argument("parent_dir", type=Path, help="Project root containing trial folders")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to MONAI checkpoint (default: model_training/lean_best.pth).",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=2,
        default=ROI_SIZE,
        help="Sliding window size, e.g. --roi-size 512 512",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PROB_THRESHOLD,
        help="Probability threshold for binary mask.",
    )
    parser.add_argument(
        "--min-object-size",
        type=int,
        default=MIN_OBJECT_SIZE,
        help="Remove connected components smaller than this many pixels.",
    )
    args = parser.parse_args()

    if args.model_path is not None:
        MODEL_PATH = args.model_path
    elif _DEFAULT_MODEL_PATH.exists():
        MODEL_PATH = _DEFAULT_MODEL_PATH
    else:
        parser.error("Missing model checkpoint. Provide --model-path.")

    ROI_SIZE = tuple(args.roi_size)
    PROB_THRESHOLD = float(args.threshold)
    MIN_OBJECT_SIZE = int(args.min_object_size)

    tasks = pipeline_utils.build_tasks(args.parent_dir)
    pipeline_utils.run_pipeline(tasks, analyze_networks)