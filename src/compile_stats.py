import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pipeline_utils
import traceback
import json
import sys
import platform
from datetime import datetime
import network_detector


def load_and_combine(input_root, output_root, suffix):
    files = list(output_root.rglob(f"*{suffix}"))
    if not files:
        return None

    dfs = []
    for p in files:
        if p.name.startswith("DATA_") or p.name.startswith("SUMMARY_") or "Combined" in p.name:
            continue

        try:
            df = pd.read_csv(p)

            analysis_root = None
            for parent in p.parents:
                if parent.name == "analysis_results":
                    analysis_root = parent
                    break

            trial_name = "Unknown"
            genotype_folder = "NA"
            if analysis_root:
                try:
                    rel_owner = analysis_root.parent.relative_to(output_root)
                    mips_owner = input_root / rel_owner
                except ValueError:
                    mips_owner = analysis_root.parent

                # could be Trial or Genotype
                has_mips_here = (mips_owner / "MIPs").exists()
                has_mips_parent = mips_owner.parent and (mips_owner.parent / "MIPs").exists()

                if has_mips_here and not has_mips_parent:
                    # Genotype-level: Trial/Genotype/{MIPs,analysis_results}
                    trial_name = mips_owner.parent.name if mips_owner.parent else "Unknown"
                    genotype_folder = mips_owner.name
                elif has_mips_here and has_mips_parent:
                    # Uncommon nested case: prefer parent as trial
                    trial_name = mips_owner.parent.name
                    genotype_folder = mips_owner.name
                elif not has_mips_here and has_mips_parent:
                    # Trial-level: Trial/{MIPs,analysis_results}
                    trial_name = mips_owner.name
                    genotype_folder = "NA"
                else:
                    # Fallback: treat analysis_results parent as trial
                    trial_name = mips_owner.name
                    genotype_folder = "NA"

            image_name = p.name.replace(f"_{suffix}", "")
            genotype = genotype_folder if genotype_folder != "NA" else "Unknown"
            group_name = trial_name

            df.insert(0, 'image_name', image_name)
            df['trial'] = trial_name
            df['folder_group'] = genotype_folder
            df['genotype'] = genotype
            df['comparison_group'] = group_name
            dfs.append(df)
        except Exception as e:
            print(f"[!] Skipped file {p}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def plot_metric_polished(ax, data, x_col, y_col, hue_col, title, ylabel, show_legend=False):
    order = sorted(data[x_col].unique())
    hue_order = sorted(data[hue_col].dropna().unique())

    if 'Norm' in title:
        ax.axhline(1.0, linestyle=':', color='black', alpha=0.4, linewidth=1)

    sns.stripplot(
        data=data,
        x=x_col, y=y_col, hue=hue_col,
        palette="tab10", order=order, hue_order=hue_order,
        dodge=True, jitter=True, size=6, alpha=0.4,
        ax=ax, legend=False, zorder=0
    )

    sns.pointplot(
        data=data,
        x=x_col, y=y_col, hue=hue_col,
        palette="tab10", order=order, hue_order=hue_order,
        dodge=0.4, capsize=0.1,
        errorbar='se', markers="D", linestyles="",
        markersize=6,
        ax=ax, legend=False, zorder=10
    )

    ax.set_title(title, fontweight='bold', fontsize=14, pad=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("")

    for i, group in enumerate(order):
        sub = data[data[x_col] == group]
        wt = sub[sub['genotype'] == 'WT'][y_col].dropna()
        mut = sub[sub['genotype'] == 'Mutant'][y_col].dropna()

        if len(wt) > 1 and len(mut) > 1:
            stat, p = ttest_ind(wt, mut, equal_var=False)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            y_max = sub[y_col].max()
            y_pos = y_max + (data[y_col].max() * 0.05)

            ax.text(i, y_pos, f"{sig}\n(p={p:.3f})",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right', frameon=False)


def make_nice_plots(net_df, output_dir):
    sns.set_theme(style="ticks", context="talk")

    plots_to_make = []

    if net_df is not None:
        plots_to_make.append((net_df, 'monai_mask_pixels', 'Network Pixels', 'Pixel Count'))

    if not plots_to_make: return

    fig1, axes = plt.subplots(len(plots_to_make), 1, figsize=(10, 5 * len(plots_to_make)), sharex=True)
    if len(plots_to_make) == 1: axes = [axes]

    for i, (data, col, title, ylab) in enumerate(plots_to_make):
        plot_metric_polished(axes[i], data, 'comparison_group', col, 'genotype', title, ylab, show_legend=(i == 0))

    plt.setp(axes[-1].get_xticklabels(), rotation=25, ha='right')
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(output_dir / "PLOT_Per_Trial_Breakdown.png", dpi=300)

    fig2, axes2 = plt.subplots(1, len(plots_to_make), figsize=(5 * len(plots_to_make), 6))
    if len(plots_to_make) == 1: axes2 = [axes2]

    for i, (data, col, title, ylab) in enumerate(plots_to_make):
        plot_metric_polished(axes2[i], data, 'genotype', col, 'genotype', f"Global: {title}", ylab, show_legend=False)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(output_dir / "PLOT_Global_Summary.png", dpi=300)
    print("[ok] plots saved.")


def save_network_montages(output_root: Path, full_res: bool):
    """create per-trial network qc montages from saved overlays."""
    print(f"\n--- generating network montages ---")
    dpi = 600 if full_res else 200
    downscale = 1.0 if full_res else 0.5
    for trial_dir in sorted([p for p in output_root.iterdir() if p.is_dir()]):
        if trial_dir.name == "analysis_runs":
            continue
        overlay_dirs = list(trial_dir.rglob("analysis_results/network_overlay"))
        images = []
        for overlay_dir in overlay_dirs:
            images.extend(sorted(overlay_dir.glob("*.tif")))
        if not images:
            continue

        output_path = trial_dir / "analysis_results" / "Montage_Network.png"
        try:
            pipeline_utils.create_montage(
                trial_dir.name,
                images,
                output_path,
                group_by_genotype=False,
                dpi=dpi,
                downscale=downscale,
            )
            print(f"[ok] {trial_dir.name}: {output_path.name}")
        except Exception:
            print(f"[!] failed montage for {trial_dir.name}")
            traceback.print_exc()


def save_global_network_montage(output_root: Path, full_res: bool):
    """create a single montage across all trials."""
    all_images = sorted([
        p for p in output_root.rglob("analysis_results/network_overlay/*.tif")
        if "analysis_runs" not in p.parts
    ])
    if not all_images:
        return

    dpi = 600 if full_res else 200
    downscale = 1.0 if full_res else 0.5

    trial_map = {}
    for p in all_images:
        try:
            rel = p.relative_to(output_root)
            trial_name = rel.parts[0] if rel.parts else "Unknown"
        except ValueError:
            trial_name = "Unknown"
        phenotype = pipeline_utils.get_genotype_from_name(p.name)
        trial_map.setdefault(trial_name, {}).setdefault(phenotype, []).append(p)

    ordered_images = []
    montage_groups = []
    trial_names = sorted(trial_map.keys())
    for i, trial_name in enumerate(trial_names):
        phenos = trial_map[trial_name]
        for phenotype in ["WT", "Mutant", "Unknown"]:
            paths = sorted(phenos.get(phenotype, []))
            ordered_images.extend(paths)
            if paths:
                color = "#555555" if phenotype == "WT" else "#E63946" if phenotype == "Mutant" else "#777777"
                montage_groups.append({
                    # "label": f"{trial_name} | {phenotype}",
                    "paths": paths,
                    "color": color,
                })
        if i < len(trial_names) - 1:
            montage_groups.append({"spacer_rows": 1})
    if not ordered_images:
        return

    output_path = output_root / "analysis_results" / "Montage_Network_All.png"
    try:
        pipeline_utils.create_montage(
            "",
            ordered_images,
            output_path,
            group_by_genotype=False,
            enhance=False,
            groups=montage_groups,
            dpi=dpi,
            downscale=downscale,
        )
        print(f"[ok] global montage: {output_path.name}")
    except Exception:
        print(f"[!] failed global montage")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-image stats and generate plots/montages."
    )
    parser.add_argument("data_dir", type=Path, help="Project root containing trial folders")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier (default: timestamp)")
    parser.add_argument("--full-res", action="store_true",
                        help="Generate full-resolution montages (slower).")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = pipeline_utils.get_output_root(args.data_dir)
    run_dir = output_root / "analysis_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- compiling stats & cleaning data in {args.data_dir} ---")
    print(f"[i] run id: {run_id}")

    net_df = load_and_combine(args.data_dir, output_root, "network_stats.csv")
    if net_df is not None:
        cols_to_save = [c for c in net_df.columns if 'Unnamed' not in c]
        net_df[cols_to_save].to_csv(run_dir / "DATA_Networks_Combined.csv", index=False)

    meta = {
        "run_id": run_id,
        "data_dir": str(args.data_dir.resolve()),
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "network_params": {
            "model_path": str(network_detector.MODEL_PATH) if network_detector.MODEL_PATH else None,
            "roi_size": network_detector.ROI_SIZE,
            "prob_threshold": network_detector.PROB_THRESHOLD,
            "min_object_size": network_detector.MIN_OBJECT_SIZE,
        },
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    save_network_montages(output_root, args.full_res)
    save_global_network_montage(output_root, args.full_res)
    make_nice_plots(net_df, run_dir)


if __name__ == "__main__":
    main()


