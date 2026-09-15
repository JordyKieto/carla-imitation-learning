"""Optional S3 storage. Enabled only when S3_BUCKET is set in .env.

boto3 picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
from the environment, so no credentials are passed around in code.
"""
import os

from utils.config import env


def bucket():
    return env("S3_BUCKET")


def enabled():
    return bucket() is not None


def _client():
    import boto3

    return boto3.client("s3", endpoint_url=env("S3_ENDPOINT_URL"))


def _require():
    if not enabled():
        raise RuntimeError("S3 is not configured. Set S3_BUCKET and AWS credentials in .env.")


def key(prefix, filename):
    return prefix.rstrip("/") + "/" + os.path.basename(filename)


def list_keys(prefix, suffixes=None):
    _require()
    paginator = _client().get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            if suffixes is None or obj["Key"].endswith(tuple(suffixes)):
                keys.append(obj["Key"])
    return keys


def download(s3_key, local_path, skip_existing=True):
    _require()
    local_path = str(local_path)
    if skip_existing and os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"[S3] s3://{bucket()}/{s3_key} -> {local_path}")
    _client().download_file(bucket(), s3_key, local_path)
    return local_path


def upload(local_path, s3_key):
    _require()
    print(f"[S3] {local_path} -> s3://{bucket()}/{s3_key}")
    _client().upload_file(str(local_path), bucket(), s3_key)
    return s3_key
