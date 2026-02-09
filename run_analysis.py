import subprocess
import sys
import argparse
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import dataset_index

def validate_inputs(data_dir: Path) -> bool:
    try:
        dataset_index.index_dataset(data_dir, require_chan1_mips=False)
    except dataset_index.DatasetLayoutError as e:
        print(f"[!] layout error: {e}")
        return False
    return True


def run_step(script_name, args):
    script_path = Path("src") / script_name
    print(f"\n>>> running {script_name}...")
    cmd = [sys.executable, str(script_path)] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"\n[!] pipeline failed at {script_name}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run network analysis and compile summary stats/plots."
    )
    parser.add_argument("dir", type=Path, help="Dataset ROOT containing trial folders")
    parser.add_argument(
        "--convert-oibs",
        "--convert_oibs",
        action="store_true",
        help="Optional: run src/convert_oibs.py before analysis",
    )
    parser.add_argument(
        "--force-convert",
        "--force_convert",
        action="store_true",
        help="With --convert-oibs: force .oib -> MIP conversion even if MIPs already exist",
    )
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to MONAI checkpoint (default: model_training/lean_best.pth).")
    parser.add_argument("--full-res", action="store_true",
                        help="Generate a single full-resolution global montage.")
    args = parser.parse_args()

    if not validate_inputs(args.dir): sys.exit(1)
    path_str = str(args.dir)

    if args.force_convert and not args.convert_oibs:
        parser.error("--force-convert requires --convert-oibs")

    if args.convert_oibs:
        convert_args = [path_str]
        if args.force_convert:
            convert_args.append("--force-convert")
        run_step("convert_oibs.py", convert_args)

    # Index for logging and final validation.
    try:
        idx = dataset_index.index_dataset(args.dir, require_chan1_mips=True)
    except dataset_index.DatasetLayoutError as e:
        print(f"[!] layout error: {e}")
        sys.exit(1)

    if not idx.images:
        print("[!] error: no channel-1 MIP TIFFs found under any trial folder.")
        print("[i] expected either ROOT/<TRIAL>/MAX_chan1_*.tif (or MIPs/chan1/*.tif)")
        print("[i] or ROOT/<TRIAL>/<GROUP>/MAX_chan1_*.tif (or MIPs/chan1/*.tif)")
        print("[i] if you only have .oib files, run: python src/convert_oibs.py <ROOT>")
        sys.exit(1)

    print("\n--- dataset discovery ---")
    for line in idx.summary_lines():
        print(line)

    net_args = [path_str]
    if args.model_path:
        net_args += ["--model-path", str(args.model_path)]
    run_step("network_detector.py", net_args)
    compile_args = [path_str]
    if args.full_res:
        compile_args.append("--montage")
    run_step("compile_stats.py", compile_args)  # Now includes montage generation

    print(f"\n[+] pipeline complete.")


if __name__ == "__main__":
    main()
