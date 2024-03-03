import os
import boto3
import configparser

#Import env variables from config path
config = configparser.ConfigParser()
config.read(r'C:\Users\payto\Desktop\Computer_Science\CS_1070\config.ini')

#S3 access keys
ACCESS_KEY = config['Credentials']['ACCESS_KEY']
SECRET_ACCESS_KEY = config['Credentials']['SECRET_ACCESS_KEY']

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY
)

def list_bucket_keys(bucket_name)->list:
    keys_in_bucket = []
    response = s3_client.list_objects_v2(Bucket=bucket_name)

    #Get first 1000 items in bucket
    if 'Contents' in response:
        for obj in response['Contents']:
            keys_in_bucket.append(obj['Key'])

    #Handle if bucket contains more than max returnable items = 1000
    while response.get('NextContinuationToken'):
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            ContinuationToken=response['NextContinuationToken']
        )
        for obj in response['Contents']:
            keys_in_bucket.append(obj['Key'])

    print(keys_in_bucket)
    return keys_in_bucket


def upload_to_s3(data_directory, bucket_name, keys_in_bucket):
    #For all files in data directory create path and upload to s3
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, data_directory)
            s3_path = os.path.join(bucket_name, relative_path).replace('\\', '/')

            if(s3_path not in keys_in_bucket):
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f'Uploaded {local_path} to S3://{bucket_name}/{s3_path}')
            else:
                print(f'{file} already exists in S3!')

data_directory = config['Data']['download_location']
bucket_name = config['Data']['bucket_name']

keys_in_bucket = list_bucket_keys(bucket_name)
upload_to_s3(data_directory, bucket_name, keys_in_bucket)