"""Shared fail-closed filesystem primitives for local datagen adapters."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from soufflerie.errors import ArtifactIntegrityError


def fsync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_real_directory(root: Path, *parts: str) -> Path:
    """Create fixed store prefixes durably and reject non-directory components."""

    current = root
    for part in parts:
        candidate = current / part
        created = False
        try:
            candidate.mkdir()
            created = True
        except FileExistsError:
            pass
        try:
            details = candidate.lstat()
        except OSError as error:
            raise ArtifactIntegrityError(
                f"unable to inspect local store directory component {part!r}"
            ) from error
        if not stat.S_ISDIR(details.st_mode):
            raise ArtifactIntegrityError(f"local store component {part!r} is not a real directory")
        if created:
            fsync_directory(candidate)
            fsync_directory(current)
        current = candidate
    return current


__all__ = ["ensure_real_directory", "fsync_directory", "fsync_file"]
