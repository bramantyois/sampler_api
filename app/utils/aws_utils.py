import os

import json

import logging
import uuid

import boto3
from botocore.exceptions import ClientError
#logger = logging.getLogger(__name__)


def upload_file(file, file_name=None):
    """Upload a file to an S3 bucket

    :param file: File to upload in binary mode
    :param file_name: S3 object name. If not specified then file_name is used
    """

    s3 = get_s3_resource()
    bucket_name = os.getenv('S3_BUCKET_NAME')
    # check if bucket exists
    bucket = get_bucket(bucket_name, s3)

    if file_name is None:
        file_name = f'{str(uuid.uuid4())}.txt'

    try:
        bucket.put_object(
            Body=file,
            Key=file_name,
            ACL='public-read'
        )
        url = f'https://{bucket_name}.s3.amazonaws.com/{file_name}'
        return url
    except Exception as e:
        return False


def upload_json(json_dict, file_name=None):
    """Upload a json file to an S3 bucket

    :param json_dict: json dict to upload
    :param file_name: S3 object name. If not specified then file_name is used
    """

    s3 = get_s3_resource()
    bucket_name = os.getenv('S3_BUCKET_NAME')
    # check if bucket exists
    bucket = get_bucket(bucket_name, s3)

    if file_name is None:
        file_name = f'{str(uuid.uuid4())}.json'

    try:
        bucket.put_object(
            Body=(bytes(json.dumps(json_dict).encode('UTF-8'))),
            Key=file_name,
            ACL='public-read'
        )
        url = f'https://{bucket_name}.s3.amazonaws.com/{file_name}'
        return url
    except Exception as e:
        return False
    

def download_file(file_key, file_name):
    s3 = get_s3_resource()
    bucket_name = os.getenv('S3_BUCKET_NAME')
    
    bucket = get_bucket(bucket_name, s3)

    try:
        bucket.download_file(
            file_key, 
            file_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def get_bucket(bucket_name, s3):
    if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
        # create bucket
        s3.create_bucket(Bucket=bucket_name)
    else:
        # get bucket
        bucket = s3.Bucket(bucket_name)
    return bucket


def get_s3_resource():
    # boto3 client
    sts = boto3.client(
        'sts',
        aws_access_key_id= os.getenv('S3_ACCESS_KEY_ID'),
        aws_secret_access_key= os.getenv('S3_SECRET_ACCESS_KEY'))
    # get credentials
    credentials = sts.assume_role(
        RoleArn= os.getenv('S3_ROLE_ARN'),
        RoleSessionName='AssumeRoleSession1'
    )['Credentials']
    # get service resource
    s3 = boto3.resource(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    return s3


