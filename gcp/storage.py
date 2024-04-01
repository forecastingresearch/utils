"""Simplify Cloud Storage interactions."""
import logging
import os
import pathlib
from os.path import basename, split

import pandas as pd

from . import storage_download_file, storage_list_files_with_prefix, storage_upload_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_with_prefix(
    bucket_name: str,
    prefix: str,
):
    """List files in the folder specified by `prefix`."""
    return storage_list_files_with_prefix.list_blobs_with_prefix(
        bucket_name=bucket_name,
        prefix=prefix,
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
        filename = basename(local_filename)
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
    if not local_filename:
        directory, basename = split(filename)
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


def get_stored_question_data(
    BUCKET_NAME,
    JSON_MARKET_FILENAME,
    LOCAL_MARKET_FILENAME,
    JSON_MARKET_VALUE_FILENAME,
    LOCAL_MARKET_VALUES_FILENAME,
):
    """Download question data from cloud storage."""
    # Initialize dataframes with predefined columns
    dfq_columns = [
        "id",
        "question",
        "background",
        "source_resolution_criteria",
        "begin_datetime",
        "close_datetime",
        "url",
        "resolved",
        "resolution_datetime",
    ]
    dfmv_columns = [
        "id",
        "datetime",
        "value",
    ]
    dfq = pd.DataFrame(columns=dfq_columns)
    dfmv = pd.DataFrame(columns=dfmv_columns)

    try:
        # Attempt to download and read the market questions file
        logger.info(f"Get questions from {BUCKET_NAME}/{JSON_MARKET_FILENAME}")
        download_no_error_message_on_404(
            bucket_name=BUCKET_NAME,
            filename=JSON_MARKET_FILENAME,
            local_filename=LOCAL_MARKET_FILENAME,
        )
        # Check if the file is not empty before reading
        if os.path.getsize(LOCAL_MARKET_FILENAME) > 0:
            dfq_tmp = pd.read_json(LOCAL_MARKET_FILENAME, lines=True)
            if not dfq_tmp.empty:
                dfq = dfq_tmp

        # Attempt to download and read the market values file
        logger.info(f"Get market values from {BUCKET_NAME}/{JSON_MARKET_VALUE_FILENAME}")
        download_no_error_message_on_404(
            bucket_name=BUCKET_NAME,
            filename=JSON_MARKET_VALUE_FILENAME,
            local_filename=LOCAL_MARKET_VALUES_FILENAME,
        )
        # Check if the file is not empty before reading
        if os.path.getsize(LOCAL_MARKET_VALUES_FILENAME) > 0:
            dfmv_tmp = pd.read_json(LOCAL_MARKET_VALUES_FILENAME, lines=True)
            if not dfmv_tmp.empty:
                dfmv = dfmv_tmp
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    return dfq, dfmv
