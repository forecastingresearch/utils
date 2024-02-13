"""Simplify Cloud Storage interactions."""

import pathlib
from os.path import basename, split

from . import storage_download_file, storage_upload_file


def upload(
    bucket_name: str,
    local_filename: str,
    destination_folder: str,
    *,
    filename: str = None,
):
    """Facilitate uploading file to cloud storage."""
    if not filename:
        filename = basename(local_filename)
    storage_upload_file.upload_blob(
        bucket_name,
        local_filename,
        f"{destination_folder}/{filename}",
    )


def download(
    bucket_name: str,
    filename: str,
    local_filename: str = None,
) -> str:
    """Facilitate downloading file from cloud storage."""
    if not local_filename:
        directory, basename = split(filename)
        local_directory = f"/tmp/{directory}"
        pathlib.Path(local_directory).mkdir(parents=True, exist_ok=True)
        local_filename = f"{local_directory}/{basename}"

    storage_download_file.download_blob(bucket_name, filename, local_filename)

    return local_filename
