import os
import requests
import time
import uuid
import json
from botocore.exceptions import NoCredentialsError, ClientError
import hashlib
from dotenv import load_dotenv

from typing import Optional, List, Literal

from pydantic import BaseModel, validator

from app.utils.aws_utils import get_s3_resource, upload_json, download_file


class BaseImage(BaseModel):
    url: str


class Image(BaseImage):
    url: str
    style_id: str
    prompt: str
    batch_idx: int
    inference_parameters: dict


class GeneratedImagesResponse(BaseModel):
    requestId: str
    images: Optional[List[BaseImage]]
    celery_task_id: Optional[str]
    user_id: Optional[str]
    status: Literal["scheduled", "completed"]


class GenerateImagesRequest(BaseModel):
    requestId: str = str(uuid.uuid4())
    async_req: bool = True
    prompt: str
    vector: Optional[List[float]]
    guidance_scale: Optional[float]
    inference_steps: Optional[int]
    num_images: Optional[int]
    seed: Optional[int]
    filenames: Optional[List[str]]

    @validator("vector", allow_reuse=True)
    def vector_length(cls, v):
        if len(v) != 768:
            raise ValueError(f"vector has length {len(v)}. Only length 768 is allowed.")
        return v


class InferenceParameters(BaseModel):
    guidance_scale: Optional[float] = 7.0
    inference_steps: Optional[int] = 50
    num_images: Optional[int] = 1
    seed: Optional[int] = 42


class ImageProvider():
    def __init__(
        self,
        style: List[int],
        prompt: str,
        inference_parameters: InferenceParameters,):
       
        self.inference_parameters = inference_parameters
        self.style = style
        self.prompt = prompt
        self.s3 = get_s3_resource()
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.backend_url = os.getenv('BACKEND_URL')
        self.file_key = self.calculate_hash_key()
        self.metadata = self.check_metadata()
        self.meta_url = None
        if self.metadata is None:
            self.schedule()

    def calculate_hash_key(self):
        hash_obj = hashlib.sha256()
        hash_obj.update(json.dumps(self.style).encode())
        hash_obj.update(self.prompt.encode())
        hash_obj.update(json.dumps(self.inference_parameters.dict()).encode())
        return hash_obj.hexdigest()

    def check_metadata(self):
        try:
            download_file(f"{self.file_key}.json", f"/tmp/{self.file_key}.json")
            
            with open(f"/tmp/{self.file_key}.json", 'r') as json_file:
                metadata_dict = json.load(json_file)

            self.meta_url = f"https://{self.bucket_name}.s3.amazonaws.com/{self.file_key}.json"
            
            return Image(**metadata_dict)
        except NoCredentialsError:
            print("No AWS credentials found.")
        except Exception as e:
            return None

    def check_image_ready(self):
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=f"{self.file_key}.png")
            return True
        except Exception as e:
            return False

    #def schedule(self, style, prompt, infer_params):
    def schedule(self,):
        if self.metadata is not None:
            print("Already scheduled, metadata exists in S3.")
            return

        request = GenerateImagesRequest(
            prompt=self.prompt, 
            vector=self.style,
            filenames=[self.file_key], 
            **self.inference_parameters.dict()
        )
        response = requests.post(
            f"{self.backend_url}/generate/test-user/",
            json=request.dict(),
        )
        print("response schedule", response)
        response = GeneratedImagesResponse.parse_raw(response.content)

        # Saving metadata to s3
        self.metadata = Image(
            url=self.get_s3_image_url(),
            style_id=self.file_key,
            prompt=self.prompt,
            batch_idx=0,
            inference_parameters=self.inference_parameters.dict(),
        )
        
        self.meta_url = upload_json(self.metadata.dict(), f"{self.file_key}.json")

        print("response schedule", response, self.file_key)

    def get_s3_image_url(self):
        return f"https://{self.bucket_name}.s3.amazonaws.com/{self.file_key}.png"

    def get_image(self):
        if not self.check_image_ready():
            print("Image is not yet ready.")
            return None
        return self.metadata

    def await_completion(self):
        while not self.check_image_ready():
            time.sleep(0.5)
        return self.get_image()


    