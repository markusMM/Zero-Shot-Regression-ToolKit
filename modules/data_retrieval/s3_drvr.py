import json
import os
import re
import tempfile
import boto3
import joblib
from modules.log import logger


# WRITE
def dump_to(
    obj,
    bucket_name,
    subdir='dumped_objects',
    key='obj.pkl'
):
    s3_client = boto3.client('s3')
    with tempfile.TemporaryFile() as fp:
        joblib.dump(obj, fp)
        fp.seek(0)
        s3_client.put_object(
            Body=fp.read(),
            Bucket=bucket_name,
            Key=os.path.join(subdir, key)
        )


# READ
def load_from(
    bucket_name,
    subdir='dumped_objects',
    key='obj.pkl'
):
    s3_client = boto3.client('s3')
    with tempfile.TemporaryFile() as fp:
        s3_client.download_fileobj(
            Fileobj=fp,
            Bucket=bucket_name,
            Key=os.path.join(subdir, key)
        )
        fp.seek(0)
        obj = joblib.load(fp)
    return obj


# DELETE
def del_obj(
    bucket_name,
    subdir='dumped_objects',
    key='obj.pkl'
):
    s3_client = boto3.client('s3')
    s3_client.delete_object(
        Bucket=bucket_name,
        Key=os.path.join(subdir, key)
    )


# LIST
def list_obj(
    bucket_name,
    subdir='dumped_objects',
    sub='\_\d+.pkl'  # noqa
):
    s3_client = boto3.resource('s3')
    my_bucket = s3_client.Bucket(bucket_name)

    objects = []
    for obj in my_bucket.objects.filter(Prefix=subdir):
        if re.search(sub, obj.key):
            objects.append(obj.key)

    return objects


def parse_path(path: str):
    path = path.split(':')[1]
    path = path.replace('//', '')
    path = path.split('/')
    return dict(
        bucket_name = path[0],
        subdir = '/'.join(path[1:-1]),
        key = path[-1]
    )


# JSON obj
def load_parameters_from_s3(
        bucket_name: str = 'cm-forecasting',
        subdir: str = 'train/config',
        key: str = 'params.json'
):
    # Load parameters from the specified S3 path (Assuming it's a JSON file)
    s3_client = boto3.client('s3')
    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=os.path.join(subdir, key)
    )
    # load and return JSON string
    return json.loads(response['Body'].read())


def upload_parameters_to_s3(
        params: dict,
        bucket_name: str = 'cm-forecasting',
        subdir: str = 'config',
        key: str = 'params.json'
):
    """
    Uploads a dictionary as a JSON file to an S3 bucket.

    Parameters:
    - params: Dictionary to be saved as JSON.
    - bucket_name: Name of the S3 bucket.
    - subdir: Subdirectory within the bucket (can be an empty string).
    - key: Key (file name) for the JSON file in the S3 bucket.
    """
    # Convert dictionary to JSON string
    json_data = json.dumps(params)

    # Create S3 client
    s3 = boto3.client('s3')

    # Upload JSON string to S3
    s3.put_object(
        Body=json_data,
        Bucket=bucket_name,
        Key=os.path.join(subdir, key)
    )


def download_dir(bucket_name, remote_directory_name, target_directory):
    if target_directory is None:
        target_directory = '.'
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix = remote_directory_name):
        try:
            new_obj = os.path.join(target_directory, obj.key)
            if not os.path.exists(os.path.dirname(new_obj)):
                os.makedirs(os.path.dirname(new_obj))
            bucket.download_file(obj.key, new_obj) # save to same path
        except Exception as e:  # noqa
            logger.warning(e)
            logger.warning(f'cannot dump {obj.key}')
