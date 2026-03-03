import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback
import pipeline_utils
from dataset_index import _strip_mip_prefix


def load_and_combine(output_root, suffix):
    files = list((output_root).rglob(f"metrics/*{suffix}"))
    if not files:
        return None

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)

            try:
                rel = p.relative_to(output_root)
            except ValueError:
                rel = None

            if rel is None or len(rel.parts) < 3:
                continue

            # Layouts supported inside ROOT/mucinet-results/:
            # - <TRIAL>/metrics/<img>_network_stats.csv
            # - <TRIAL>/<GROUP>/metrics/<img>_network_stats.csv
            trial_name = rel.parts[0]
            if rel.parts[1] == "metrics":
                comparison_group = "NA"
            else:
                comparison_group = rel.parts[1]
                if len(rel.parts) < 4 or rel.parts[2] != "metrics":
                    continue

            image_name = p.name.replace(f"_{suffix}", "")
            image_id = _strip_mip_prefix(image_name)

            df.insert(0, "image_id", image_id)
            df.insert(0, "comparison_group", comparison_group)
            df.insert(0, "is_wt", comparison_group == "WT")
            df.insert(0, "trial_name", trial_name)
            dfs.append(df)
        except Exception as e:
            print(f"[!] Skipped file {p}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def make_bar_plots(net_df, output_dir: Path):
    sns.set_theme(style="ticks", context="talk")

    if net_df is None or net_df.empty:
        return

    df = net_df.copy()
    df["comparison_group"] = df["comparison_group"].fillna("NA")

    y_col = "monai_mask_pixels"
    y_label = "Pixel Count"
    title_suffix = ""
    if "monai_mask_pixels_norm_to_wt" in df.columns and df["monai_mask_pixels_norm_to_wt"].notna().any():
        y_col = "monai_mask_pixels_norm_to_wt"
        y_label = "Norm Pixel Count"
        title_suffix = " (Normalized to WT)"
        df = df[df[y_col].notna()].copy()

    # Per-trial bar plot (means with SE).
    fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.75 * df["trial_name"].nunique()), 7))
    sns.barplot(
        data=df,
        x="trial_name",
        y=y_col,
        hue="comparison_group",
        errorbar="se",
        ax=ax,
    )
    ax.set_title(f"Network Pixels Per Trial{title_suffix}", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    if y_col == "monai_mask_pixels_norm_to_wt":
        ax.axhline(1.0, linestyle=":", color="black", alpha=0.4, linewidth=1)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    for lab in ax.get_xticklabels():
        lab.set_ha("center")
    fig.subplots_adjust(bottom=0.35)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(output_dir / "per_trial_bar.png", dpi=300)
    plt.close(fig)

    # Global bar plot (means with SE).
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    sns.barplot(
        data=df,
        x="comparison_group",
        y=y_col,
        errorbar="se",
        ax=ax2,
    )
    ax2.set_title(f"Network Pixels Global{title_suffix}", fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel(y_label)
    if y_col == "monai_mask_pixels_norm_to_wt":
        ax2.axhline(1.0, linestyle=":", color="black", alpha=0.4, linewidth=1)
    ax2.tick_params(axis="x", labelrotation=90, labelsize=10)
    for lab in ax2.get_xticklabels():
        lab.set_ha("center")
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(output_dir / "global_bar.png", dpi=300)
    plt.close(fig2)

    print("[ok] plots saved.")


def save_global_network_montage(output_root: Path):
    """Create a global montage of all network overlay images."""
    all_images = sorted(list(output_root.rglob("network_overlay/*.tif")) + list(output_root.rglob("network_overlay/*.tiff")))
    if not all_images:
        return

    trial_map = {}
    for p in all_images:
        rel = None
        try:
            rel = p.relative_to(output_root)
            trial_name = rel.parts[0] if rel.parts else "Unknown"
        except ValueError:
            trial_name = "Unknown"
        group_name = "Unknown"
        if rel is not None:
            # ROOT/mucinet-results/<TRIAL>/network_overlay/*.tif
            # ROOT/mucinet-results/<TRIAL>/<GROUP>/network_overlay/*.tif
            if len(rel.parts) >= 3 and rel.parts[1] == "network_overlay":
                group_name = "NA"
            elif len(rel.parts) >= 4 and rel.parts[2] == "network_overlay":
                group_name = rel.parts[1]
        trial_map.setdefault(trial_name, {}).setdefault(group_name, []).append(p)

    ordered_images = []
    montage_groups = []
    trial_names = sorted(trial_map.keys())
    for i, trial_name in enumerate(trial_names):
        phenos = trial_map[trial_name]
        group_names = sorted(phenos.keys(), key=lambda x: (x != "WT", x))
        for group_name in group_names:
            paths = sorted(phenos.get(group_name, []))
            ordered_images.extend(paths)
            if paths:
                color = "#555555" if group_name == "WT" else "#E63946" if group_name != "NA" else "#777777"
                montage_groups.append({
                    "paths": paths,
                    "color": color,
                })
        if i < len(trial_names) - 1:
            montage_groups.append({"spacer_rows": 1})
    if not ordered_images:
        return

    output_path = output_root / "global_montage.png"
    try:
        pipeline_utils.create_montage(
            ordered_images,
            output_path,
            groups=montage_groups,
        )
        print(f"[ok] global montage: {output_path.name}")
    except Exception:
        print(f"[!] failed global montage")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-image stats and generate plots."
    )
    parser.add_argument("--dir", type=Path, required=True, help="Dataset root containing trial folders")
    parser.add_argument("--montage", action="store_true",
                        help="Generate a single full-resolution global montage.")
    args = parser.parse_args()

    output_root = pipeline_utils.get_output_root(args.dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"--- compiling stats & cleaning data in {args.dir} ---")

    net_df = load_and_combine(output_root, "network_stats.csv")
    if net_df is not None:
        # Normalize each trial to its own WT mean so images from different sessions are comparable.
        # We do this per-trial rather than globally because illumination varies between sessions.
        if "comparison_group" in net_df.columns and "monai_mask_pixels" in net_df.columns:
            net_df["monai_mask_pixels_norm_to_wt"] = pd.NA
            for trial_name, idxs in net_df.groupby("trial_name").groups.items():
                sub = net_df.loc[idxs]
                has_groups = (sub["comparison_group"] != "NA").any()
                has_wt = (sub["comparison_group"] == "WT").any()
                if not has_groups or not has_wt:
                    continue
                wt_mean = sub.loc[sub["comparison_group"] == "WT", "monai_mask_pixels"].mean()
                if wt_mean and wt_mean > 0:
                    net_df.loc[idxs, "monai_mask_pixels_norm_to_wt"] = sub["monai_mask_pixels"] / wt_mean

        # pandas sometimes inserts "Unnamed: 0" columns when round-tripping through CSV; drop them
        cols_to_save = [c for c in net_df.columns if 'Unnamed' not in c]
        net_df[cols_to_save].to_csv(output_root / "combined_metrics.csv", index=False)

    if args.montage:
        save_global_network_montage(output_root)

    make_bar_plots(net_df, output_root)


if __name__ == "__main__":
    main()
