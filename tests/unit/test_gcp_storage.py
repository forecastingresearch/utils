"""Tests for GCP storage utilities - local mount path and local root tests."""

import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import NotFound

from utils.gcp import storage as gcp_storage
from utils.gcp import storage_list_files_with_prefix
from utils.gcp.storage import (
    download,
    download_no_error_message_on_404,
    file_exists,
    get_last_modified_time,
    list,
    list_with_prefix,
    upload,
)


def test_list_with_mount_directory():
    """Test listing files using local mount directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock bucket structure
        bucket_name = "test-bucket"
        mount_dir = os.path.join(tmpdir, bucket_name)
        os.makedirs(mount_dir, exist_ok=True)

        # Create test files in root
        file1 = os.path.join(mount_dir, "file1.txt")
        file2 = os.path.join(mount_dir, "file2.txt")
        with open(file1, "w") as f:
            f.write("content1")
        with open(file2, "w") as f:
            f.write("content2")

        # Create test files in subdirectory
        subdir = os.path.join(mount_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        file3 = os.path.join(subdir, "file3.txt")
        with open(file3, "w") as f:
            f.write("content3")

        # Test listing with mount directory
        result = list(bucket_name=bucket_name, mnt=tmpdir)

        # Verify all files are listed with relative paths
        assert len(result) == 3
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir/file3.txt" in result


def test_get_last_modified_time_with_mount_directory():
    """Test getting last modified time using local mount directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_name = "test-bucket"
        mount_dir = os.path.join(tmpdir, bucket_name)
        os.makedirs(mount_dir, exist_ok=True)

        # Create a test file
        filename = "test_file.txt"
        filepath = os.path.join(mount_dir, filename)
        with open(filepath, "w") as f:
            f.write("test content")

        # Get the actual mtime from the filesystem
        expected_mtime = os.path.getmtime(filepath)

        # Test getting last modified time
        result = get_last_modified_time(bucket_name=bucket_name, filename=filename, mnt=tmpdir)

        # Verify result is a datetime object
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

        # Verify the timestamp matches (within 1 second tolerance)
        result_timestamp = result.timestamp()
        assert abs(result_timestamp - expected_mtime) < 1


# Tests for list_blobs_with_prefix function
class TestListBlobsWithPrefix:
    """Test cases for storage_list_files_with_prefix.list_blobs_with_prefix."""

    @patch("utils.gcp.storage_list_files_with_prefix.storage.Client")
    def test_list_blobs_with_prefix_success(self, mock_client_class):
        """Test successfully listing blobs with a prefix."""
        # Create mock blob objects
        mock_blob1 = MagicMock()
        mock_blob1.name = "folder1/file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = "folder1/file2.txt"
        mock_blob3 = MagicMock()
        mock_blob3.name = "folder2/file3.txt"

        # Set up the mock client
        mock_client = MagicMock()
        mock_client.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
        mock_client_class.return_value = mock_client

        # Call the function
        result = storage_list_files_with_prefix.list_blobs_with_prefix(
            bucket_name="test-bucket", prefix="folder1/"
        )

        # Verify results
        assert len(result) == 3
        assert "folder1/file1.txt" in result
        assert "folder1/file2.txt" in result
        assert "folder2/file3.txt" in result

        # Verify the client was called correctly
        mock_client_class.assert_called_once()
        mock_client.list_blobs.assert_called_once_with(
            "test-bucket", prefix="folder1/", delimiter=None
        )
        mock_client.close.assert_called_once()


# Tests for the GCP_STORAGE_LOCAL_ROOT opt-in: a local directory tree stands in for a bucket.
LOCAL_ROOT_ENV_VAR = "GCP_STORAGE_LOCAL_ROOT"
BUCKET = "test-bucket"


def _write(path: Path, text: str) -> Path:
    """Create `path` (and its parents) with the given text and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def local_bucket(tmp_path, monkeypatch):
    """Point GCP_STORAGE_LOCAL_ROOT at a temporary root holding an empty local copy of BUCKET."""
    root = tmp_path / "gcs-root"
    bucket_dir = root / BUCKET
    bucket_dir.mkdir(parents=True)
    monkeypatch.setenv(LOCAL_ROOT_ENV_VAR, str(root))
    return bucket_dir


@pytest.fixture
def gcs_unreachable(monkeypatch):
    """Replace every route to GCS with a mock and fail the test if any of them was used.

    `download_no_error_message_on_404` and `get_last_modified_time` swallow exceptions, so
    the check runs after the test body rather than relying on the mocks raising.
    """
    mocks = {}
    targets = [
        (gcp_storage.storage, "Client"),
        (gcp_storage.storage_list_files, "list_blobs"),
        (gcp_storage.storage_list_files_with_prefix, "list_blobs_with_prefix"),
        (gcp_storage.storage_upload_file, "upload_blob"),
        (gcp_storage.storage_download_file, "download_blob"),
    ]
    for module, name in targets:
        mock = MagicMock(side_effect=AssertionError(f"GCS was contacted via {name}"))
        monkeypatch.setattr(module, name, mock)
        mocks[name] = mock
    yield mocks
    for name, mock in mocks.items():
        assert not mock.called, f"GCS was contacted via {name}"


def test_local_root_list_returns_sorted_relative_names(local_bucket, gcs_unreachable):
    """With a local root, `list` walks the bucket directory and never contacts GCS."""
    _write(local_bucket / "questions" / "b.json", "b")
    _write(local_bucket / "questions" / "a.json", "a")
    _write(local_bucket / "top.txt", "t")

    assert list(bucket_name=BUCKET) == ["questions/a.json", "questions/b.json", "top.txt"]


def test_local_root_is_read_at_call_time(tmp_path, monkeypatch):
    """Setting GCP_STORAGE_LOCAL_ROOT after the module is imported is honoured on the next call."""
    monkeypatch.delenv(LOCAL_ROOT_ENV_VAR, raising=False)
    root = tmp_path / "gcs-root"
    _write(root / BUCKET / "local.txt", "local")

    with patch.object(gcp_storage.storage_list_files, "list_blobs", return_value=["remote.txt"]):
        assert list(bucket_name=BUCKET) == ["remote.txt"]

        monkeypatch.setenv(LOCAL_ROOT_ENV_VAR, str(root))
        assert list(bucket_name=BUCKET) == ["local.txt"]


@pytest.mark.parametrize("root_exists", [True, False])
def test_local_root_without_bucket_directory_fails_closed(
    tmp_path, monkeypatch, gcs_unreachable, root_exists
):
    """Once the root is set, a missing `<root>/<bucket>` raises and never reaches GCS."""
    root = tmp_path / "gcs-root"
    if root_exists:
        (root / "some-other-bucket").mkdir(parents=True)
    monkeypatch.setenv(LOCAL_ROOT_ENV_VAR, str(root))
    source = _write(tmp_path / "source.txt", "payload")
    target = str(tmp_path / "downloaded.txt")

    calls = [
        lambda: list(bucket_name=BUCKET),
        lambda: list_with_prefix(bucket_name=BUCKET, prefix="remote/"),
        lambda: file_exists(bucket_name=BUCKET, filename="remote.txt"),
        lambda: upload(bucket_name=BUCKET, local_filename=str(source), destination_folder="f"),
        lambda: download(bucket_name=BUCKET, filename="remote.txt", local_filename=target),
        lambda: get_last_modified_time(bucket_name=BUCKET, filename="remote.txt"),
    ]
    for call in calls:
        with pytest.raises(FileNotFoundError) as excinfo:
            call()
        assert LOCAL_ROOT_ENV_VAR in str(excinfo.value)
        assert str(root / BUCKET) in str(excinfo.value)

    # The tolerant wrapper swallows the error as it does any other, but still never reaches GCS.
    tolerant = download_no_error_message_on_404(
        bucket_name=BUCKET, filename="remote.txt", local_filename=target
    )
    assert tolerant == target

    assert not (root / BUCKET).exists()
    assert not os.path.exists(target)


@pytest.mark.parametrize("form", ["home", "relative"])
def test_local_root_expands_home_and_resolves_relative_paths(
    tmp_path, monkeypatch, gcs_unreachable, form
):
    """A `~`-prefixed or relative root is normalized to the absolute directory it names."""
    root = tmp_path / "gcs-root"
    _write(root / BUCKET / "local.txt", "local")
    if form == "home":
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(LOCAL_ROOT_ENV_VAR, "~/gcs-root")
    else:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(LOCAL_ROOT_ENV_VAR, "gcs-root")

    assert list(bucket_name=BUCKET) == ["local.txt"]
    with pytest.raises(FileNotFoundError) as excinfo:
        list(bucket_name="unknown-bucket")
    assert str(root / "unknown-bucket") in str(excinfo.value)


def test_mnt_without_bucket_directory_falls_through_to_local_root(
    local_bucket, tmp_path, gcs_unreachable
):
    """`mnt` stays fail-open: a missing `<mnt>/<bucket>` defers to the environment root."""
    _write(local_bucket / "from_env.txt", "env")
    mnt = tmp_path / "mnt-without-bucket"
    mnt.mkdir()

    assert list(bucket_name=BUCKET, mnt=str(mnt)) == ["from_env.txt"]
    result = get_last_modified_time(bucket_name=BUCKET, filename="from_env.txt", mnt=str(mnt))
    assert isinstance(result, datetime)


def test_explicit_mnt_takes_precedence_over_local_root(local_bucket, tmp_path, gcs_unreachable):
    """`list` and `get_last_modified_time` use the `mnt` tree when both it and the root exist."""
    _write(local_bucket / "from_env.txt", "env")
    mnt = tmp_path / "mnt"
    mnt_file = _write(mnt / BUCKET / "from_mnt.txt", "mnt")

    assert list(bucket_name=BUCKET, mnt=str(mnt)) == ["from_mnt.txt"]
    assert list(bucket_name=BUCKET) == ["from_env.txt"]

    result = get_last_modified_time(bucket_name=BUCKET, filename="from_mnt.txt", mnt=str(mnt))
    assert abs(result.timestamp() - os.path.getmtime(mnt_file)) < 1
    assert get_last_modified_time(bucket_name=BUCKET, filename="from_env.txt", mnt=str(mnt)) is None


@pytest.mark.parametrize("escape", ["absolute", "../{name}", "nested/../../{name}"])
def test_local_root_rejects_object_names_that_escape_the_bucket(
    local_bucket, tmp_path, gcs_unreachable, escape
):
    """An absolute name or a `..` segment never reads or writes outside the bucket directory."""
    outside = tmp_path / "outside.txt"
    name = str(outside) if escape == "absolute" else escape.format(name="outside.txt")
    source = _write(tmp_path / "staging" / "payload.txt", "payload")

    calls = [
        lambda: file_exists(bucket_name=BUCKET, filename=name),
        lambda: upload(bucket_name=BUCKET, local_filename=str(source), filename=name),
        lambda: download(bucket_name=BUCKET, filename=name, local_filename=str(tmp_path / "d")),
        lambda: get_last_modified_time(bucket_name=BUCKET, filename=name),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()

    assert not outside.exists()
    assert list(bucket_name=BUCKET) == []


def test_local_root_rejects_empty_object_name(local_bucket, tmp_path, gcs_unreachable):
    """An empty name would denote the bucket directory itself, which is not an object."""
    for call in (
        lambda: file_exists(bucket_name=BUCKET, filename=""),
        lambda: download(bucket_name=BUCKET, filename="", local_filename=str(tmp_path / "d")),
        lambda: get_last_modified_time(bucket_name=BUCKET, filename=""),
    ):
        with pytest.raises(ValueError):
            call()


def test_local_root_upload_rejects_escaping_destination_folder(
    local_bucket, tmp_path, gcs_unreachable
):
    """`destination_folder` is part of the object name and is confined the same way."""
    source = _write(tmp_path / "staging" / "payload.txt", "payload")

    with pytest.raises(ValueError):
        upload(bucket_name=BUCKET, local_filename=str(source), destination_folder="..")

    assert not (tmp_path / "gcs-root" / "payload.txt").exists()
    assert list(bucket_name=BUCKET) == []


def test_local_root_list_with_prefix_matches_object_name_prefix(local_bucket, gcs_unreachable):
    """`list_with_prefix` is a plain string prefix on object names, like GCS, with sorted output."""
    _write(local_bucket / "forecasts" / "2024-01-02.json", "")
    _write(local_bucket / "forecasts" / "2024-01-01.json", "")
    _write(local_bucket / "forecasts-archive" / "old.json", "")
    _write(local_bucket / "questions" / "q.json", "")

    assert list_with_prefix(bucket_name=BUCKET, prefix="forecasts/") == [
        "forecasts/2024-01-01.json",
        "forecasts/2024-01-02.json",
    ]
    assert list_with_prefix(bucket_name=BUCKET, prefix="forecasts") == [
        "forecasts-archive/old.json",
        "forecasts/2024-01-01.json",
        "forecasts/2024-01-02.json",
    ]
    assert list_with_prefix(bucket_name=BUCKET, prefix="nothing-here") == []


def test_local_root_file_exists_is_an_exact_object_name_check(local_bucket, gcs_unreachable):
    """`file_exists` matches only the full object name, not prefixes or directories."""
    _write(local_bucket / "data" / "report.json", "{}")

    assert file_exists(bucket_name=BUCKET, filename="data/report.json") is True
    assert file_exists(bucket_name=BUCKET, filename="data/report") is False
    assert file_exists(bucket_name=BUCKET, filename="data") is False
    assert file_exists(bucket_name=BUCKET, filename="report.json") is False


def test_local_root_upload_without_destination_folder(local_bucket, tmp_path, gcs_unreachable):
    """`upload` stores the object under its basename at the bucket root and overwrites in place."""
    source = _write(tmp_path / "staging" / "data.csv", "v1")

    upload(bucket_name=BUCKET, local_filename=str(source))
    assert (local_bucket / "data.csv").read_text() == "v1"

    source.write_text("v2")
    upload(bucket_name=BUCKET, local_filename=str(source))
    assert (local_bucket / "data.csv").read_text() == "v2"
    assert list(bucket_name=BUCKET) == ["data.csv"]


def test_local_root_upload_with_destination_folder_and_filename(
    local_bucket, tmp_path, gcs_unreachable
):
    """`upload` stores the object at `<destination_folder>/<filename>`, creating the folders."""
    source = _write(tmp_path / "staging" / "data.csv", "payload")

    upload(bucket_name=BUCKET, local_filename=str(source), destination_folder="a/b")
    upload(
        bucket_name=BUCKET,
        local_filename=str(source),
        destination_folder="a/b",
        filename="renamed.csv",
    )

    assert (local_bucket / "a" / "b" / "data.csv").read_text() == "payload"
    assert (local_bucket / "a" / "b" / "renamed.csv").read_text() == "payload"
    assert list(bucket_name=BUCKET) == ["a/b/data.csv", "a/b/renamed.csv"]
    assert file_exists(bucket_name=BUCKET, filename="a/b/renamed.csv") is True


def test_local_root_upload_sets_modified_time_to_upload_time(
    local_bucket, tmp_path, gcs_unreachable
):
    """An uploaded object's modified time is when it was uploaded, not the source file's mtime."""
    source = _write(tmp_path / "old.csv", "payload")
    long_ago = datetime(2000, 1, 1, tzinfo=timezone.utc)
    os.utime(source, (long_ago.timestamp(), long_ago.timestamp()))

    upload(bucket_name=BUCKET, local_filename=str(source))

    result = get_last_modified_time(bucket_name=BUCKET, filename="old.csv")
    assert result.tzinfo == timezone.utc
    assert result > long_ago + timedelta(days=1)
    assert abs(result - datetime.now(tz=timezone.utc)) < timedelta(minutes=5)


def test_local_root_download_to_default_path(local_bucket, gcs_unreachable):
    """`download` without `local_filename` copies the object to `/tmp/<dir>/<basename>`."""
    directory = f"gcs-local-root-test-{uuid.uuid4().hex}"
    _write(local_bucket / directory / "data.txt", "payload")

    try:
        result = download(bucket_name=BUCKET, filename=f"{directory}/data.txt")
        assert result == f"/tmp/{directory}/data.txt"
        assert Path(result).read_text() == "payload"
    finally:
        shutil.rmtree(f"/tmp/{directory}", ignore_errors=True)


def test_local_root_download_to_explicit_path(local_bucket, tmp_path, gcs_unreachable):
    """`download` with `local_filename` copies the object there and returns that path."""
    _write(local_bucket / "data" / "file.txt", "payload")
    target = tmp_path / "out" / "copy.txt"
    target.parent.mkdir()

    assert download(
        bucket_name=BUCKET, filename="data/file.txt", local_filename=str(target)
    ) == str(target)
    assert target.read_text() == "payload"


def test_local_root_download_missing_object_raises_not_found(
    local_bucket, tmp_path, gcs_unreachable
):
    """A missing object raises `NotFound`, like the GCS client, instead of falling back to GCS."""
    target = tmp_path / "copy.txt"

    with pytest.raises(NotFound):
        download(bucket_name=BUCKET, filename="missing.txt", local_filename=str(target))
    assert not target.exists()


def test_local_root_download_no_error_message_on_404_missing_object(
    local_bucket, tmp_path, gcs_unreachable
):
    """The 404-tolerant wrapper returns the target path for a missing object without raising."""
    target = tmp_path / "copy.txt"

    result = download_no_error_message_on_404(
        bucket_name=BUCKET, filename="missing.txt", local_filename=str(target)
    )

    assert result == str(target)
    assert not target.exists()


def test_local_root_get_last_modified_time_missing_object_returns_none(
    local_bucket, gcs_unreachable
):
    """`get_last_modified_time` returns None for an object that is not in the local tree."""
    _write(local_bucket / "present.txt", "x")

    assert get_last_modified_time(bucket_name=BUCKET, filename="missing.txt") is None
    assert get_last_modified_time(bucket_name=BUCKET, filename="present.txt") is not None
