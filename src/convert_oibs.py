import argparse
from pathlib import Path
import numpy as np
import oiffile
import tifffile
from tqdm import tqdm


def convert_single_oib(oib_path: Path, owner_dir: Path):
    # Use context manager to safely handle file opening
    with oiffile.OifFile(oib_path) as oib:
        image = oib.asarray()

    if image.shape[0] < 2:
        raise ValueError(f"Unexpected OIB array shape (need at least 2 planes in first dim), got shape={image.shape}")

    # Single-channel max projection saved next to the OIB.
    chan = image[1]
    save_path = owner_dir / f"MAX_{oib_path.stem}.tif"
    max_z = np.max(chan, axis=0)
    tifffile.imwrite(save_path, max_z)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Convert OIBs to Max-Z TIFs")
    parser.add_argument("dir", type=Path, help="Dataset ROOT containing trial folders")
    parser.add_argument(
        "--force-convert",
        "--force_convert",
        action="store_true",
        help="Recompute max projection TIFFs even if outputs already exist",
    )
    args = parser.parse_args()

    # Only process OIBs located directly inside phenotype folders: ROOT/<TRIAL>/<PHENOTYPE>/*.oib
    all_oibs = list(args.dir.rglob("*.oib"))
    oib_files = []
    for p in all_oibs:
        try:
            rel = p.relative_to(args.dir)
        except ValueError:
            continue
        if "mucinet-results" in rel.parts:
            continue
        if len(rel.parts) != 3:
            continue
        oib_files.append(p)

    if not oib_files:
        print(f"No .oib files found in {args.dir}")
        return

    print(f"Found {len(oib_files)} OIB files. Converting...")

    for oib_path in tqdm(oib_files, desc="Converting"):
        try:
            owner_dir = oib_path.parent
            expected1 = owner_dir / f"MAX_{oib_path.stem}.tif"

            if not args.force_convert and expected1.exists():
                continue

            convert_single_oib(oib_path, owner_dir)
        except Exception as e:
            print(f"Error converting {oib_path.name}: {e}")

    print("\n[!] Conversion Complete.")



if __name__ == "__main__":
    main()
