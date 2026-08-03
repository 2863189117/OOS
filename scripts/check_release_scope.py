#!/usr/bin/env python3
"""Fail when tracked files cross the OOS public-release boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKER = Path(__file__).resolve()
BANNED_DIRECTORIES = {
    "archive",
    "data",
    "g4",
    "output",
    "paper",
    "results",
    "runs",
    "tmp",
}
BANNED_EXTENSIONS = {
    ".csv",
    ".dat",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".log",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".root",
    ".tif",
    ".tiff",
    ".tsv",
}
BANNED_SCRIPT_PREFIXES = (
    "analyze_",
    "build_adaptive_fisher_",
    "compare_response",
    "plot_",
    "run_exact_commit_",
)
BANNED_BASENAMES = {
    ".nature-figure.json",
}
INTERNAL_MARKERS = (
    "/home/",
    "/Users/",
    "".join(("XL", "ZD")),
    "".join(("xl", "zd")),
    "".join(("D", "OR")),
    "".join(("include/", "d", "or", "/")),
    "".join(("namespace ", "d", "or")),
    "".join(("d", "or", "::")),
    "".join(("lib", "d", "or")),
    "".join(("--", "d", "or")),
    "".join(("center-s", "2")),
    "".join(("active_radius_mm: float = 14", "90.0")),
    "".join(("wall_radius_mm: float = 15", "60.0")),
    "".join(("lxe_depth_mm: float = 39", "65.0")),
    "".join(("pmt_count\": 10", "66")),
)


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        ROOT / field.decode()
        for field in completed.stdout.split(b"\0")
        if field
    ]


def main() -> int:
    violations: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            violations.append(f"symbolic link is not allowed: {relative}")
        if any(part in BANNED_DIRECTORIES for part in relative.parts):
            violations.append(f"out-of-scope directory: {relative}")
        if path.suffix.lower() in BANNED_EXTENSIONS:
            violations.append(f"generated/data extension: {relative}")
        if path.name in BANNED_BASENAMES or path.name.startswith(
            BANNED_SCRIPT_PREFIXES
        ):
            violations.append(f"post-processing/internal file: {relative}")
        if path == CHECKER or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"unexpected binary file: {relative}")
            continue
        for marker in INTERNAL_MARKERS:
            if marker in content:
                violations.append(
                    f"internal path or host marker {marker!r}: {relative}"
                )
    if violations:
        print("OOS release-scope check failed:", file=sys.stderr)
        for violation in sorted(set(violations)):
            print(f"  - {violation}", file=sys.stderr)
        return 1
    print("OOS release-scope check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
