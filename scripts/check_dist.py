#!/usr/bin/env python3
"""Validate release archive metadata and contents."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile


PACKAGE = "mlx_kfac"
DIST_NAME = "mlx-kfac"


def _safe_members(names: set[str], archive: Path) -> None:
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise AssertionError(f"unsafe member in {archive.name}: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_wheel(wheel: Path, version: str) -> None:
    dist_info = f"{PACKAGE}-{version}.dist-info"
    package_sources = {
        f"{PACKAGE}/{path.name}"
        for path in Path("src", PACKAGE).glob("*.py")
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        _safe_members(names, wheel)
        required = package_sources | {
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/RECORD",
            f"{dist_info}/licenses/LICENSE",
        }
        missing = required - names
        if missing:
            raise AssertionError(
                f"{wheel.name} is missing: {sorted(missing)}"
            )
        if any("/tests/" in f"/{name}" or name.startswith("tests/") for name in names):
            raise AssertionError(f"{wheel.name} must not contain the test suite")
        metadata = BytesParser().parsebytes(
            archive.read(f"{dist_info}/METADATA")
        )

    assert metadata["Name"] == DIST_NAME
    assert metadata["Version"] == version
    assert metadata["License-Expression"] == "MIT"
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert metadata["Requires-Python"] == ">=3.10"
    requirements = metadata.get_all("Requires-Dist", [])
    mlx_requirement = next(
        (item.replace(" ", "") for item in requirements if item.startswith("mlx")),
        "",
    )
    assert ">=0.25.2" in mlx_requirement and "<0.33" in mlx_requirement
    assert {"dev", "release", "test"} <= set(
        metadata.get_all("Provides-Extra", [])
    )
    assert len(metadata.get_all("Project-URL", [])) >= 5


def check_sdist(sdist: Path, version: str) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
    _safe_members(names, sdist)
    roots = {PurePosixPath(name).parts[0] for name in names if name}
    if len(roots) != 1:
        raise AssertionError(f"{sdist.name} has unexpected roots: {roots}")
    root = roots.pop()
    source_python_files = {
        f"{root}/{path.as_posix()}"
        for directory in (Path("scripts"), Path("src", PACKAGE), Path("tests"))
        for path in directory.rglob("*.py")
    }
    required = {
        f"{root}/CHANGELOG.md",
        f"{root}/LICENSE",
        f"{root}/MANIFEST.in",
        f"{root}/README.md",
        f"{root}/RELEASING.md",
        f"{root}/pyproject.toml",
    } | source_python_files
    missing = required - names
    if missing:
        raise AssertionError(f"{sdist.name} is missing: {sorted(missing)}")
    if root != f"{PACKAGE}-{version}":
        raise AssertionError(
            f"unexpected sdist root {root!r}; expected {PACKAGE}-{version!s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dist",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="directory containing exactly one wheel and one sdist",
    )
    args = parser.parse_args()
    config = tomllib.loads(Path("pyproject.toml").read_text())
    version = config["project"]["version"]
    wheels = sorted(args.dist.glob("*.whl"))
    sdists = sorted(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"expected one wheel and one sdist in {args.dist}; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    expected_stem = f"{PACKAGE}-{version}"
    if not wheels[0].name.startswith(expected_stem):
        raise AssertionError(f"wheel version does not match {version}")
    if not sdists[0].name.startswith(expected_stem):
        raise AssertionError(f"sdist version does not match {version}")
    check_wheel(wheels[0], version)
    check_sdist(sdists[0], version)
    for artifact in (*wheels, *sdists):
        print(f"{_sha256(artifact)}  {artifact.name}")


if __name__ == "__main__":
    main()
