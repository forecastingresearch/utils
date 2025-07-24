"""Compression utilities: create and extract tar.gz archives."""

import os
import shutil
import tarfile
from typing import List


def compress(files: List[str], archive_name: str) -> str:
    """
    Compress multiple files into a single tar.gz archive.

    Args:
        files: List of paths to the files to include.
        archive_basename: Full path where the archive will be created.

    Returns:
        The path to the created archive.
    """
    with tarfile.open(archive_name, "w:gz") as archive:
        for filepath in files:
            archive.add(filepath, arcname=os.path.basename(filepath))
    return archive_name


def extract(
    archive_name: str,
    extract_dir: str,
    rm_dir_before_extract: str = "",
    rm_archive_on_extract: bool = False,
) -> str:
    """
    Extract a tar.gz archive into a directory.

    Args:
        archive_basename: Full path of the archive to extract.
        extract_dir: Directory where files will be unpacked.

    Returns:
        The path to the extraction directory.
    """
    if rm_dir_before_extract and os.path.isdir(rm_dir_before_extract):
        shutil.rmtree(rm_dir_before_extract)

    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_name, "r:gz") as archive:
        archive.extractall(path=extract_dir)

    if rm_archive_on_extract:
        os.remove(archive_name)

    return extract_dir
