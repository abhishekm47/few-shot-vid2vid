import boto3, botocore

import sys

sys.path.append('/home/ubuntu/making_with_ml/ai_dubs')

from dubber import dub
from Wav2Lip.inference import wav2lip_inference_modify

s3 = boto3.client(
   "s3",
   aws_access_key_id='AKIAWCUNX4JVZ5YCZNDX',
   aws_secret_access_key='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',
   region_name= 'us-east-2'
)

bucket = "babble-mobile-camera-uploads-ios-vid-prod"
local_path = "/home/ubuntu/few-shot-vid2vid/dataset/raw/"

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket)

relevant_set = []



for page in pages:
    for obj in page['Contents']:
        if obj['Key'].startswith("pre_2_a_"):

            relevant_set.append(obj['Key'])


keys = relevant_set[:100]

for key in keys:
    s3.download_file(bucket,key,local_path+key)

