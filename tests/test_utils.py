import os 

import json

from dotenv import load_dotenv

from app.utils.aws_utils import download_file, upload_json

import random


load_dotenv('hf.env')
load_dotenv('aws.env')
load_dotenv('backend.env')

def test_upload_download():
    # some random number
    rand_num = random.randint(0, 1000)

    # create a random json file
    file_path_down = '/tmp/test_down.json'
    file_key = 'test.json'
    file_content = {
        'test': 'test-1-2-3',
        'rand_num': rand_num
    }

    # upload file
    url = upload_json(file_content, file_key)

    # download file
    download_file(file_key, file_path_down)

    # check if file exists
    assert os.path.exists(file_path_down)

    # check if file content is the same
    with open(file_path_down, 'r') as json_file:
        file_content_down = json.load(json_file)
    
    assert file_content == file_content_down
