"""Tests for GCP storage utilities - local mount path tests."""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from utils.gcp import storage_list_files_with_prefix
from utils.gcp.storage import get_last_modified_time, list


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
