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
    # Matches current MIP naming from src/convert_oibs.py: MAX_chanX_<image_id>.tif
    for prefix in ("MAX_chan0_", "MAX_chan1_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _list_chan1_tiffs(owner_dir: Path) -> List[Path]:
    """Return chan1 MIP tiffs for an owner directory, preferring MIPs/chan1 if present."""
    mip_dir = owner_dir / "MIPs" / "chan1"
    if mip_dir.exists():
        mips = sorted([p for p in mip_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")])
        if mips:
            return mips
    # Newer conversion output: MAX_chan1_*.tif next to OIBs
    tifs = sorted(list(owner_dir.glob("MAX_chan1_*.tif")) + list(owner_dir.glob("MAX_chan1_*.tiff")))
    return tifs


@dataclass(frozen=True)
class DatasetImage:
    root: Path
    trial_name: str
    comparison_group: Optional[str]  # e.g. "WT" or other; None when trial has no comparison folders
    image_id: str  # derived from file stem

    trial_dir: Path
    owner_dir: Path  # either trial_dir or trial_dir/<comparison_group>

    mip_chan1_path: Path
    mip_chan0_path: Optional[Path]
    source_oib_path: Optional[Path]

    output_root: Path
    derived_mask_path: Path
    derived_overlay_path: Path
    derived_metrics_csv_path: Path


@dataclass(frozen=True)
class TrialIndex:
    trial_name: str
    trial_dir: Path
    groups_present: Tuple[str, ...]  # empty when trial-level MIPs (no comparison groups)
    image_counts: Dict[str, int]  # group -> count; uses "NA" when trial-level


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
    require_chan1_mips: bool = True,
) -> DatasetIndex:
    """
    Index datasets rooted at ROOT with trial folders as immediate children.

    Each trial may be either:
    - Trial-level: ROOT/<TRIAL>/MIPs/chan1/*.tif
    - Grouped:     ROOT/<TRIAL>/<GROUP>/MIPs/chan1/*.tif

    WT is treated specially by downstream stats only when a group folder named exactly "WT" exists.
    """
    trial_dirs = _iter_trial_dirs(root)

    output_root = root / "mucinet-results"

    trials_out: List[TrialIndex] = []
    images_out: List[DatasetImage] = []

    for trial_dir in trial_dirs:
        # Trial-level MIPs
        trial_mips = _list_chan1_tiffs(trial_dir)

        # Grouped MIPs (one level down)
        group_dirs = [p for p in sorted(trial_dir.iterdir()) if p.is_dir() and p.name != "MIPs"]
        group_mips_by_name: Dict[str, List[Path]] = {}
        for gd in group_dirs:
            mips = _list_chan1_tiffs(gd)
            if mips:
                group_mips_by_name[gd.name] = mips

        if trial_mips and group_mips_by_name:
            raise DatasetLayoutError(
                f"Mixed trial layout detected: {trial_dir} contains both trial-level MIPs and grouped MIPs. "
                f"Use either ROOT/<TRIAL>/MIPs/... or ROOT/<TRIAL>/<GROUP>/MIPs/... per trial."
            )

        counts: Dict[str, int] = {}

        if trial_mips:
            counts["NA"] = 0
            for mip in trial_mips:
                image_id = _strip_mip_prefix(mip.stem)

                chan0_path = trial_dir / "MIPs" / "chan0" / f"MAX_chan0_{image_id}.tif"
                mip_chan0 = chan0_path if chan0_path.exists() else None

                oib_path = (trial_dir / f"{image_id}.oib") if (trial_dir / f"{image_id}.oib").exists() else None

                rel_owner = trial_dir.relative_to(root)
                results_root = output_root / rel_owner
                derived_mask = results_root / "network_binary" / mip.name
                derived_overlay = results_root / "network_overlay" / mip.name
                derived_metrics = results_root / "metrics" / f"{mip.stem}_network_stats.csv"

                images_out.append(
                    DatasetImage(
                        root=root,
                        trial_name=trial_dir.name,
                        comparison_group=None,
                        image_id=image_id,
                        trial_dir=trial_dir,
                        owner_dir=trial_dir,
                        mip_chan1_path=mip,
                        mip_chan0_path=mip_chan0,
                        source_oib_path=oib_path,
                        output_root=output_root,
                        derived_mask_path=derived_mask,
                        derived_overlay_path=derived_overlay,
                        derived_metrics_csv_path=derived_metrics,
                    )
                )
                counts["NA"] += 1

            trials_out.append(
                TrialIndex(
                    trial_name=trial_dir.name,
                    trial_dir=trial_dir,
                    groups_present=tuple(),
                    image_counts=counts,
                )
            )
            continue

        # Grouped trial
        groups_present = tuple(sorted(group_mips_by_name.keys()))
        for group_name, mips in group_mips_by_name.items():
            if require_chan1_mips and not mips:
                continue
            counts[group_name] = counts.get(group_name, 0) + len(mips)

            owner_dir = trial_dir / group_name
            for mip in mips:
                image_id = _strip_mip_prefix(mip.stem)

                chan0_path = owner_dir / "MIPs" / "chan0" / f"MAX_chan0_{image_id}.tif"
                mip_chan0 = chan0_path if chan0_path.exists() else None

                oib_path = (owner_dir / f"{image_id}.oib") if (owner_dir / f"{image_id}.oib").exists() else None

                rel_owner = owner_dir.relative_to(root)
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
                        owner_dir=owner_dir,
                        mip_chan1_path=mip,
                        mip_chan0_path=mip_chan0,
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
