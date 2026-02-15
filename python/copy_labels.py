#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil


def parse_args() -> argparse.Namespace:
    default_src = Path(
        "/Users/pszyc/Library/CloudStorage/GoogleDrive-przemek.7678@gmail.com/My Drive/Studia/Ogniska"
    )
    default_dst = Path(__file__).resolve().parents[1] / "data"

    parser = argparse.ArgumentParser(
        description="Copy labels.json files to a local data folder, preserving relative paths."
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
        "--dry-run", action="store_true", help="Show what would be copied"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copying when destination file already exists",
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

    copied = 0
    skipped = 0
    missing = 0

    candidates = [p for p in src_root.rglob(args.pattern) if p.is_dir()]

    for candidate in candidates:
        labels_path = candidate / "labels.json"
        if not labels_path.exists():
            missing += 1
            continue

        relative = candidate.relative_to(src_root)
        out_dir = dst_root / relative
        out_labels = out_dir / "labels.json"

        if args.skip_existing and out_labels.exists():
            skipped += 1
            if args.dry_run:
                print(f"[DRY-RUN] skip existing: {out_labels}")
            continue

        if args.dry_run:
            print(f"[DRY-RUN] {labels_path} -> {out_labels}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(labels_path, out_labels)
        copied += 1

    print("Done")
    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"labels.json copied: {copied}")
    print(f"labels.json missing in candidates: {missing}")
    print(f"labels.json skipped (already exist): {skipped}")


if __name__ == "__main__":
    main()
