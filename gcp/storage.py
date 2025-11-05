"""Simplify Cloud Storage interactions."""

import os
import pathlib
from datetime import datetime, timezone
from typing import List, Optional


def list_with_prefix(
    bucket_name: str,
    prefix: str,
):
    """List files in the folder specified by `prefix`."""
    from . import storage_list_files_with_prefix

    return storage_list_files_with_prefix.list_blobs_with_prefix(
        bucket_name=bucket_name,
        prefix=prefix,
    )


def list(
    bucket_name: str,
    mnt: str = "",
) -> List[str]:
    """List files in the bucket.

    Args:
        bucket_name (str): name of bucket.
        mnt (str): mount dir of bucket.

    Returns:
        List[str]: list of files in bucket.
    """
    mount_dir = f"{mnt}/{bucket_name}"
    if mnt and os.path.exists(mount_dir):
        retval = []
        for root, _, files in os.walk(mount_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, mount_dir)
                retval.append(rel_path)
        return retval

    from . import storage_list_files

    return storage_list_files.list_blobs(
        bucket_name=bucket_name,
    )


def upload(
    bucket_name: str,
    local_filename: str,
    destination_folder: str = "",
    *,
    filename: str = None,
):
    """Facilitate uploading file to cloud storage."""
    from . import storage_upload_file

    if not filename:
        filename = os.path.basename(local_filename)
    destination_filename = f"{destination_folder}/{filename}" if destination_folder else filename

    storage_upload_file.upload_blob(
        bucket_name,
        local_filename,
        destination_filename,
    )


def download(
    bucket_name: str,
    filename: str,
    local_filename: str = None,
) -> str:
    """Facilitate downloading file from cloud storage."""
    from . import storage_download_file

    if not local_filename:
        directory, basename = os.path.split(filename)
        local_directory = f"/tmp/{directory}"
        pathlib.Path(local_directory).mkdir(parents=True, exist_ok=True)
        local_filename = f"{local_directory}/{basename}"

    storage_download_file.download_blob(bucket_name, filename, local_filename)

    return local_filename


def download_no_error_message_on_404(
    bucket_name: str,
    filename: str,
    local_filename: str = None,
) -> str:
    """Wrap `download()` function that doesn't print "Error" message if requested file not found."""
    try:
        local_filename = download(
            bucket_name=bucket_name,
            filename=filename,
            local_filename=local_filename,
        )
    except Exception:
        print(f"GCP Storage: could not download {bucket_name}/{filename}.")

    return local_filename


def get_last_modified_time(
    bucket_name: str,
    filename: str,
    mnt: str = "",
) -> Optional[datetime]:
    """Return the last modified date for the given file."""
    from google.cloud import storage

    mount_dir = f"{mnt}/{bucket_name}"
    if mnt and os.path.exists(mount_dir):
        ts = os.path.getmtime(f"{mount_dir}/{filename}")
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.get_blob(filename)
        return blob.updated if blob else None
    except Exception:
        print(f"GCP Storage: could not find modified time of {bucket_name}/{filename}.")
        return None
