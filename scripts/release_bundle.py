#!/usr/bin/env python3
"""Create portable release bundles under dist/."""

from __future__ import annotations

import hashlib
import tarfile
import tomllib
import zipfile
from pathlib import Path

EXCLUDES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist"}


def is_excluded(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in EXCLUDES for part in parts)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    name = pyproject["project"]["name"]
    version = pyproject["project"]["version"]

    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    base = f"{name}-{version}"
    tar_path = dist / f"{base}.tar.gz"
    zip_path = dist / f"{base}.zip"

    with tarfile.open(tar_path, "w:gz") as tar:
        for p in root.rglob("*"):
            if p.is_dir() or is_excluded(p, root):
                continue
            tar.add(p, arcname=f"{base}/{p.relative_to(root)}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_dir() or is_excluded(p, root):
                continue
            zf.write(p, arcname=f"{base}/{p.relative_to(root)}")

    checksum = dist / f"{base}.sha256"
    checksum.write_text(
        f"{sha256_of(tar_path)}  {tar_path.name}\n{sha256_of(zip_path)}  {zip_path.name}\n",
        encoding="utf-8",
    )

    print(f"Created: {tar_path}")
    print(f"Created: {zip_path}")
    print(f"Created: {checksum}")


if __name__ == "__main__":
    main()
