from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

FOV_X_UM = 90.0
FOV_Y_UM = 67.2
FOV_Z_UM = 15.0

X_MIN = 4
X_MAX = 18


def get_voxel_volume_um3(mask_shape_zyx):
    z_px, y_px, x_px = mask_shape_zyx
    sx_um = FOV_X_UM / float(x_px)
    sy_um = FOV_Y_UM / float(y_px)
    sz_um = FOV_Z_UM / float(z_px)
    return sx_um * sy_um * sz_um


def parse_args() -> argparse.Namespace:
    default_src = Path(
        "/Users/pszyc/Library/CloudStorage/GoogleDrive-przemek.7678@gmail.com/My Drive/Studia/Ogniska"
    )
    default_dst = Path(__file__).resolve().parents[1] / "data"

    parser = argparse.ArgumentParser(
        description="Copy paired foci_mask.npy and nuclei_mask.npy to local data folder."
    )
    parser.add_argument(
        "--src", type=Path, default=default_src, help="Source root directory"
    )
    parser.add_argument(
        "--dst", type=Path, default=default_dst, help="Destination root directory"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="00*",
        help="Glob pattern for candidate image folders (default: 00*)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_root}")

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    copied_dirs = 0
    copied_files = 0
    skipped_candidates = 0
    skipped_existing = 0

    def validate_uint8_compatible(
        arr: np.ndarray, arr_name: str, arr_path: Path
    ) -> None:
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))

        if not np.isfinite(arr_min) or not np.isfinite(arr_max):
            raise ValueError(f"{arr_name} contains non-finite values at {arr_path}")

        if arr_min < 0 or arr_max > 255:
            raise ValueError(
                f"{arr_name} out of uint8 range at {arr_path}: min={arr_min}, max={arr_max}"
            )

    candidates = [p for p in src_root.rglob(args.pattern) if p.is_dir()]

    for candidate in tqdm(candidates, desc="Copying mask pairs", unit="dir"):

        foci_path = candidate / "foci_mask.npy"
        nuclei_path = candidate / "nuclei_mask.npy"

        if not (foci_path.exists() and nuclei_path.exists()):
            skipped_candidates += 1
            continue

        relative_dir = candidate.relative_to(src_root)
        out_dir = dst_root / relative_dir
        out_foci = out_dir / "foci_mask.npy"
        out_nuclei = out_dir / "nuclei_mask.npy"

        if args.skip_existing and out_foci.exists() and out_nuclei.exists():
            skipped_existing += 1
            if args.dry_run:
                print(f"[DRY-RUN] skip existing: {out_foci}")
                print(f"[DRY-RUN] skip existing: {out_nuclei}")
            continue

        if args.dry_run:
            print(f"[DRY-RUN] {foci_path} -> {out_foci}")
            print(f"[DRY-RUN] {nuclei_path} -> {out_nuclei}")
            if args.overwrite_cloud:
                print(f"[DRY-RUN] overwrite source: {foci_path}")
                print(f"[DRY-RUN] overwrite source: {nuclei_path}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            foci_arr = np.load(foci_path)
            nuclei_arr = np.load(nuclei_path)

            if foci_arr.ndim != 4:
                raise ValueError(
                    f"foci_mask.npy must be 4D (N/Z/Y/X), got shape {foci_arr.shape} at {foci_path}"
                )
            if nuclei_arr.ndim != 3:
                raise ValueError(
                    f"nuclei_mask.npy must be 3D (Z/Y/X), got shape {nuclei_arr.shape} at {nuclei_path}"
                )
            if foci_arr.shape[1:] != nuclei_arr.shape:
                raise ValueError(
                    f"Spatial shape mismatch between foci and nuclei at {candidate}: "
                    f"foci.shape[1:]={foci_arr.shape[1:]} != nuclei.shape={nuclei_arr.shape}"
                )

            validate_uint8_compatible(foci_arr, "foci_mask", foci_path)
            validate_uint8_compatible(nuclei_arr, "nuclei_mask", nuclei_path)

            np.save(out_foci, foci_arr.astype(np.uint8, copy=False))
            np.save(out_nuclei, nuclei_arr.astype(np.uint8, copy=False))

        copied_dirs += 1
        copied_files += 2

    print("Done")
    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Matched directories copied: {copied_dirs}")
    print(f"Files copied: {copied_files}")
    print(f"Candidates skipped (missing one/both masks): {skipped_candidates}")
    print(f"Directories skipped (already exist locally): {skipped_existing}")
    print(f"Skip existing: {args.skip_existing}")


if __name__ == "__main__":
    main()
