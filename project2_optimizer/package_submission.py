"""Create a clean submission zip excluding caches and nested stale git metadata."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "project2_optimizer_submission.zip"

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_FILE_SUFFIXES = {".pyc", ".pyo"}
STALE_DUPLICATE_ROOT = "Automated-Optimizer-Selection-System"
SUBMISSION_ZIP_NAMES = {"project2_optimizer_submission.zip"}


def should_skip(path: Path, relative: Path) -> bool:
    if relative.parts and relative.parts[0] == STALE_DUPLICATE_ROOT:
        return True
    if any(part in SKIP_DIR_NAMES for part in relative.parts):
        return True
    if path.suffix.lower() in SKIP_FILE_SUFFIXES:
        return True
    if relative.name in SUBMISSION_ZIP_NAMES:
        return True
    posix = relative.as_posix()
    if posix.startswith(".git/") or "/.git/" in posix or posix.endswith("/.git"):
        return True
    return False


def build_archive(output: Path) -> int:
    count = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if should_skip(path, relative):
                continue
            archive.write(path, arcname=Path("project2_optimizer") / relative)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Package project2_optimizer for submission.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination zip path",
    )
    args = parser.parse_args()
    files = build_archive(args.output)
    print(f"Wrote {args.output} ({files} files)")


if __name__ == "__main__":
    main()
