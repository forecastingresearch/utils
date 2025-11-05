"""Tests for archiving utilities."""

import os
import tarfile
import tempfile

from utils.archiving.tar_gz import compress, extract


def test_compress_multiple_files():
    """Test compressing multiple files into a single tar.gz archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(tmpdir, f"test_{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"Content {i}")
            test_files.append(test_file)

        # Compress them
        archive_path = os.path.join(tmpdir, "test.tar.gz")
        result = compress(test_files, archive_path)

        # Verify the archive was created
        assert result == archive_path
        assert os.path.exists(archive_path)

        # Verify the archive contains all files
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getnames()
            assert len(members) == 3
            assert "test_0.txt" in members
            assert "test_1.txt" in members
            assert "test_2.txt" in members


def test_extract_archive():
    """Test extracting a tar.gz archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(tmpdir, f"test_{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"Content {i}")
            test_files.append(test_file)

        # Create an archive
        archive_path = os.path.join(tmpdir, "test.tar.gz")
        compress(test_files, archive_path)

        # Extract the archive
        extract_dir = os.path.join(tmpdir, "extracted")
        result = extract(archive_path, extract_dir)

        # Verify the extraction directory was created
        assert result == extract_dir
        assert os.path.isdir(extract_dir)

        # Verify all files were extracted with correct content
        for i in range(3):
            extracted_file = os.path.join(extract_dir, f"test_{i}.txt")
            assert os.path.exists(extracted_file)
            with open(extracted_file, "r") as f:
                assert f.read() == f"Content {i}"


def test_extract_with_rm_dir_before_extract():
    """Test extracting with rm_dir_before_extract parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = []
        for i in range(2):
            test_file = os.path.join(tmpdir, f"test_{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"Content {i}")
            test_files.append(test_file)

        # Create an archive
        archive_path = os.path.join(tmpdir, "test.tar.gz")
        compress(test_files, archive_path)

        # Create a directory with some files that should be removed
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        old_file = os.path.join(extract_dir, "old_file.txt")
        with open(old_file, "w") as f:
            f.write("This should be removed")

        # Extract with rm_dir_before_extract
        extracted_dir = extract(archive_path, extract_dir, rm_dir_before_extract=extract_dir)
        assert extracted_dir == extract_dir

        # Verify old file is gone and new files are extracted
        assert not os.path.exists(old_file)
        assert os.path.exists(os.path.join(extract_dir, "test_0.txt"))
        assert os.path.exists(os.path.join(extract_dir, "test_1.txt"))


def test_extract_with_rm_archive_on_extract():
    """Test extracting with rm_archive_on_extract parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = []
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Content")
        test_files.append(test_file)

        # Create an archive
        archive_path = os.path.join(tmpdir, "test.tar.gz")
        compress(test_files, archive_path)

        # Extract with rm_archive_on_extract
        extract_dir = os.path.join(tmpdir, "extracted")
        _ = extract(archive_path, extract_dir, rm_archive_on_extract=True)

        # Verify archive was removed
        assert not os.path.exists(archive_path)
        # Verify files were extracted
        assert os.path.exists(os.path.join(extract_dir, "test.txt"))
