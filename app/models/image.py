import os
import datetime
import requests
import time
import uuid
import json
import boto3
import hashlib
from typing import Optional, List, Literal

from pydantic import BaseModel

from models.request import Request
from models.response import Response


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


class ImageProvider(BaseModel):
    def __init__(
        self,
        style: List[int],
        prompt: str,
        infer_params: InferenceParameters,
    ):
        self.infer_params = infer_params
        self.style = style
        self.prompt = prompt
        self.s3 = boto3.client("s3")
        self.bucket_name = "your-bucket-name"
        self.file_key = self.calculate_hash_key()
        self.metadata = self.check_metadata()
        if self.metadata is None:
            self.schedule(style, prompt, self.infer_params)

    def calculate_hash_key(self):
        hash_obj = hashlib.sha256()
        hash_obj.update(json.dumps(self.style).encode())
        hash_obj.update(self.prompt.encode())
        hash_obj.update(json.dumps(self.infer_params.dict()).encode())
        return hash_obj.hexdigest()

    def check_metadata(self):
        try:
            self.s3.download_file(
                self.bucket_name, f"{self.file_key}.json", "/tmp/metadata.json"
            )
            with open("/tmp/metadata.json") as f:
                metadata_dict = json.load(f)
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

    def schedule(self, style, prompt, infer_params):
        if self.metadata is not None:
            print("Already scheduled, metadata exists in S3.")
            return

        request = GenerateImagesRequest(
            prompt=prompt, vector=style, **infer_params.dict()
        )
        response = requests.post(
            f"{BACKEND_URL}/generate/test-user/",
            json=request.dict(),
        )
        print("response schedule", response)
        response = GeneratedImagesResponse.parse_raw(response.content)

        # Saving metadata to s3
        self.metadata = Image(
            url=self.get_s3_image_url(),
            style_id=self.file_key,
            prompt=prompt,
            batch_idx=0,
            inference_parameters=infer_params.dict(),
        )
        with open("/tmp/metadata.json", "w") as f:
            json.dump(self.metadata.dict(), f)
        self.s3.upload_file(
            "/tmp/metadata.json", self.bucket_name, f"{self.file_key}.json"
        )

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
