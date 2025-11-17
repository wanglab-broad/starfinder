""" 
    This snakemake file includes Google Cloud backup workflows
"""

### ==================== [ Basic setting ] =========================

import os 
import glob
import random
import pandas as pd
from pathlib import Path
from google.cloud import storage

PROCESSED_DIR = "/home/unix/jiahao/wanglab/Data/Processed"
PROCESSED_BUCKET = 'wanglab-data-processed'

# Instantiates a client
storage_client = storage.Client()

bucket = storage_client.get_bucket(PROCESSED_BUCKET)
blobs = bucket.list_blobs()
target_files = [blob.name for blob in blobs if blob.name[0].isdigit()]
query_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.zip')]
files_to_upload = [f for f in query_files if f not in target_files]
files_to_upload = ['test.zip', 'sample-dataset.zip']

### ==================== [ Helper functions ] =========================

def upload_file_with_transfer_manager(bucket_name, source_file_name, destination_blob_name): 
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    import datetime
    t_set = lambda: datetime.datetime.now().astimezone().replace(microsecond=0)
    t_diff = lambda t: str(t_set() - t)

    t = t_set()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(f"File {source_file_name} uploaded to {bucket_name}.")
    print(f"Time elapsed: {t_diff(t)}")

### ==================== [ Rules ] =========================

rule all:
    input:

rule upload_to_gcs:
    input:
        expand("{input_dir}/{file_name}", input_dir=PROCESSED_DIR, file_name=files_to_upload),
    output:
        expand("{input_dir}/{file_name}", input_dir=PROCESSED_DIR, file_name=files_to_upload),
    resources:
        mem_mb=8000,
        runtime=600
    # conda:
    #     "gcloud-backup"
    run:
        upload_file_with_transfer_manager(PROCESSED_BUCKET, input[0], input[0].split('/')[-1])