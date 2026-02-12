from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DatasetLayoutError(RuntimeError):
    """Raised when the dataset root does not match the required layout."""


def _iter_trial_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise DatasetLayoutError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise DatasetLayoutError(f"Dataset root is not a directory: {root}")
    trial_dirs = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir() and p.name not in ("mucinet-results",)
        ]
    )
    if not trial_dirs:
        raise DatasetLayoutError(f"Dataset root has no trial folders: {root}")
    return trial_dirs


def _strip_mip_prefix(stem: str) -> str:
    # Matches max-projection naming conventions: MAX_<image_id>.tif (and legacy MAX_chan*_...)
    for prefix in ("MAX_", "MAX_chan1_", "MAX_chan0_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _list_input_tiffs(owner_dir: Path) -> List[Path]:
    """
    Return input TIFFs located directly in the owner directory.
    Only TIFFs directly inside the phenotype folder are used.
    """
    return sorted(list(owner_dir.glob("*.tif")) + list(owner_dir.glob("*.tiff")))


@dataclass(frozen=True)
class DatasetImage:
    root: Path
    trial_name: str
    comparison_group: Optional[str]  # e.g. "WT" or other; None when trial has no comparison folders
    image_id: str  # derived from file stem

    trial_dir: Path
    owner_dir: Path  # either trial_dir or trial_dir/<comparison_group>

    tiff_path: Path
    source_oib_path: Optional[Path]

    output_root: Path
    derived_mask_path: Path
    derived_overlay_path: Path
    derived_metrics_csv_path: Path


@dataclass(frozen=True)
class TrialIndex:
    trial_name: str
    trial_dir: Path
    groups_present: Tuple[str, ...]  # phenotype subdirectory names
    image_counts: Dict[str, int]  # phenotype -> count


@dataclass(frozen=True)
class DatasetIndex:
    root: Path
    trials: Tuple[TrialIndex, ...]
    images: Tuple[DatasetImage, ...]

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        lines.append(f"[i] trials discovered: {len(self.trials)}")
        for t in self.trials:
            if t.groups_present:
                g_list = ", ".join(t.groups_present)
                lines.append(f"[i] {t.trial_name}: groups={g_list}")
                for g in t.groups_present:
                    lines.append(f"[i] {t.trial_name} / {g}: images={t.image_counts.get(g, 0)}")
            else:
                lines.append(f"[i] {t.trial_name}: no comparison groups")
                lines.append(f"[i] {t.trial_name}: images={t.image_counts.get('NA', 0)}")
        return lines


def index_dataset(
    root: Path,
    *,
    require_tiffs: bool = True,
) -> DatasetIndex:
    """
    Index datasets rooted at ROOT with trial folders as immediate children.

    Analysis inputs are expected at ROOT/<TRIAL>/<PHENOTYPE>/*.tif(f).
    Only TIFFs directly inside phenotype folders are used.
    """
    trial_dirs = _iter_trial_dirs(root)

    output_root = root / "mucinet-results"

    trials_out: List[TrialIndex] = []
    images_out: List[DatasetImage] = []

    for trial_dir in trial_dirs:
        # Trial-level TIFFs are not accepted; TIFFs must live in phenotype folders.
        trial_mips = _list_input_tiffs(trial_dir)
        if trial_mips:
            raise DatasetLayoutError(
                f"Invalid trial layout: found TIFFs directly under {trial_dir}. "
                f"Expected ROOT/<TRIAL>/<PHENOTYPE>/*.tif"
            )

        # Consider only phenotype dirs that contain at least one TIFF or OIB directly inside them.
        phenotype_dirs = []
        for p in sorted(trial_dir.iterdir()):
            if not p.is_dir():
                continue
            has_direct_tiff = any(p.glob("*.tif")) or any(p.glob("*.tiff"))
            has_direct_oib = any(p.glob("*.oib"))
            if has_direct_tiff or has_direct_oib:
                phenotype_dirs.append(p)
        if not phenotype_dirs:
            raise DatasetLayoutError(
                f"Invalid trial folder (missing phenotype subdirs): {trial_dir}. "
                f"Expected ROOT/<TRIAL>/<PHENOTYPE>/*.tif"
            )

        counts: Dict[str, int] = {p.name: 0 for p in phenotype_dirs}
        groups_present = tuple([p.name for p in phenotype_dirs])

        for phenotype_dir in phenotype_dirs:
            group_name = phenotype_dir.name
            mips = _list_input_tiffs(phenotype_dir)
            if require_tiffs and not mips:
                continue

            counts[group_name] += len(mips)
            for mip in mips:
                image_id = _strip_mip_prefix(mip.stem)

                oib_path = (phenotype_dir / f"{image_id}.oib") if (phenotype_dir / f"{image_id}.oib").exists() else None

                rel_owner = phenotype_dir.relative_to(root)
                results_root = output_root / rel_owner
                derived_mask = results_root / "network_binary" / mip.name
                derived_overlay = results_root / "network_overlay" / mip.name
                derived_metrics = results_root / "metrics" / f"{mip.stem}_network_stats.csv"

                images_out.append(
                    DatasetImage(
                        root=root,
                        trial_name=trial_dir.name,
                        comparison_group=group_name,
                        image_id=image_id,
                        trial_dir=trial_dir,
                        owner_dir=phenotype_dir,
                        tiff_path=mip,
                        source_oib_path=oib_path,
                        output_root=output_root,
                        derived_mask_path=derived_mask,
                        derived_overlay_path=derived_overlay,
                        derived_metrics_csv_path=derived_metrics,
                    )
                )

        trials_out.append(
            TrialIndex(
                trial_name=trial_dir.name,
                trial_dir=trial_dir,
                groups_present=groups_present,
                image_counts=counts,
            )
        )

    return DatasetIndex(root=root, trials=tuple(trials_out), images=tuple(images_out))
