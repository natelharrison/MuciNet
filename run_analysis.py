import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


def validate_inputs(data_dir: Path) -> bool:
    print(f"--- checking inputs in: {data_dir} ---")
    images = list(data_dir.rglob("MIPs/chan1/*.tif"))

    if not images:
        print("[!] error: missing images.")
        return False

    print(f"[+] found {len(images)} images.")
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
    parser.add_argument("dir", type=Path, help="Project root containing trial folders")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional run identifier; defaults to timestamp")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to MONAI checkpoint (default: model_training/lean_best.pth).")
    parser.add_argument("--full-res", action="store_true",
                        help="Generate full-resolution montages (slower).")
    args = parser.parse_args()

    if not validate_inputs(args.dir): sys.exit(1)
    path_str = str(args.dir)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    net_args = [path_str]
    if args.model_path:
        net_args += ["--model-path", str(args.model_path)]
    run_step("network_detector.py", net_args)
    compile_args = [path_str, "--run-id", run_id]
    if args.full_res:
        compile_args.append("--full-res")
    run_step("compile_stats.py", compile_args)  # Now includes montage generation

    print(f"\n[+] pipeline complete.")


if __name__ == "__main__":
    main()
