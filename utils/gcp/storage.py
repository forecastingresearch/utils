"""Simplify Cloud Storage interactions."""

import os
import pathlib
import shutil
import tempfile
from datetime import datetime, timezone
from typing import List, Optional

from google.api_core.exceptions import NotFound
from google.cloud import storage

from . import (
    storage_download_file,
    storage_list_files,
    storage_list_files_with_prefix,
    storage_upload_file,
)

# Offline opt-in: serve every bucket from ``<root>/<bucket>`` instead of GCS. Unset in deployment.
LOCAL_ROOT_ENV_VAR = "GCP_STORAGE_LOCAL_ROOT"


def _local_root() -> str:
    """Return the local root as an absolute path, or "" when unset. Read on every call."""
    root = os.environ.get(LOCAL_ROOT_ENV_VAR, "")
    return os.path.abspath(os.path.expanduser(root)) if root else ""


def _local_bucket_dir(bucket_name: str, mnt: str = "") -> Optional[str]:
    """Return the local directory standing in for ``bucket_name``, or None to use GCS.

    An explicit ``mnt`` wins when ``<mnt>/<bucket>`` exists; Cloud Run mounts only some buckets,
    so a missing mount is not an error. The environment root fails closed: a missing
    ``<root>/<bucket>`` raises rather than reaching GCS with real credentials.
    """
    if mnt:
        bucket_dir = os.path.join(mnt, bucket_name)
        if os.path.isdir(bucket_dir):
            return bucket_dir

    root = _local_root()
    if not root:
        return None
    bucket_dir = os.path.join(root, bucket_name)
    if not os.path.isdir(bucket_dir):
        raise FileNotFoundError(
            f"{LOCAL_ROOT_ENV_VAR} is set to {root!r} but {bucket_dir!r} is not a directory. "
            f"Create it to serve bucket {bucket_name!r} locally, or unset {LOCAL_ROOT_ENV_VAR} "
            "to use GCS."
        )
    return bucket_dir


def _local_object_path(bucket_dir: str, name: str) -> str:
    """Return the on-disk path of object ``name``, refusing names that escape ``bucket_dir``."""
    bucket_dir = os.path.abspath(bucket_dir)
    path = os.path.abspath(os.path.join(bucket_dir, name))
    if path == bucket_dir or os.path.commonpath([bucket_dir, path]) != bucket_dir:
        raise ValueError(
            f"object name {name!r} resolves outside the local bucket directory {bucket_dir!r}"
        )
    return path


def _walk_relative(bucket_dir: str) -> List[str]:
    """Return every file under ``bucket_dir`` as a sorted bucket-relative path."""
    retval = []
    for root, _, files in os.walk(bucket_dir):
        for filename in files:
            retval.append(os.path.relpath(os.path.join(root, filename), bucket_dir))
    return sorted(retval)


def list_with_prefix(
    bucket_name: str,
    prefix: str,
):
    """List files in the folder specified by `prefix`."""
    bucket_dir = _local_bucket_dir(bucket_name)
    if bucket_dir is not None:
        # GCS semantics: a plain string prefix on the object name, not a directory.
        return [name for name in _walk_relative(bucket_dir) if name.startswith(prefix)]

    return storage_list_files_with_prefix.list_blobs_with_prefix(
        bucket_name=bucket_name,
        prefix=prefix,
    )


def file_exists(bucket_name: str, filename: str) -> bool:
    """Return True if an object with this exact name exists in the bucket."""
    bucket_dir = _local_bucket_dir(bucket_name)
    if bucket_dir is not None:
        return os.path.isfile(_local_object_path(bucket_dir, filename))

    storage_client = storage.Client()
    return storage_client.bucket(bucket_name).blob(filename).exists()


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
    bucket_dir = _local_bucket_dir(bucket_name, mnt)
    if bucket_dir is not None:
        return _walk_relative(bucket_dir)

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
    if not filename:
        filename = os.path.basename(local_filename)
    destination_filename = f"{destination_folder}/{filename}" if destination_folder else filename

    bucket_dir = _local_bucket_dir(bucket_name)
    if bucket_dir is not None:
        dest = _local_object_path(bucket_dir, destination_filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Atomic replace: a reader sees the old object or the complete new one, as with GCS.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".tmp")
        os.close(fd)
        try:
            # copyfile, not copy2: the modification time is the upload time, like ``blob.updated``.
            shutil.copyfile(local_filename, tmp)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return

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
    if not local_filename:
        directory, basename = os.path.split(filename)
        local_directory = f"/tmp/{directory}"
        pathlib.Path(local_directory).mkdir(parents=True, exist_ok=True)
        local_filename = f"{local_directory}/{basename}"

    bucket_dir = _local_bucket_dir(bucket_name)
    if bucket_dir is not None:
        source_path = _local_object_path(bucket_dir, filename)
        if not os.path.isfile(source_path):
            # The local tree is authoritative: raise the client's ``NotFound``, never fall through.
            raise NotFound(f"{bucket_name}/{filename} not found under {bucket_dir}")
        shutil.copy2(source_path, local_filename)
        return local_filename

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
    bucket_dir = _local_bucket_dir(bucket_name, mnt)
    if bucket_dir is not None:
        path = _local_object_path(bucket_dir, filename)
        if not os.path.isfile(path):
            return None  # no object, no modification time
        return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.get_blob(filename)
        return blob.updated if blob else None
    except Exception:
        print(f"GCP Storage: could not find modified time of {bucket_name}/{filename}.")
        return None
