import os 

from dotenv import load_dotenv

import numpy as np
from PIL import Image as PILImage

import datasets

from app.models.image import Image, InferenceParameters, ImageProvider, GenerateImagesRequest, GeneratedImagesResponse

load_dotenv('hf.env')
load_dotenv('aws.env')
load_dotenv('backend.env')

hf_dataset = os.getenv("HF_DATASET")
hf_dataset = datasets.load_dataset(hf_dataset)

embeddings = hf_dataset["train"]["embedding"]
embeddings = np.vstack(embeddings)

def test_image():
    prompt_1 = "A painting of a dog in the style "
    prompt_2 = "A painting of a cat in the style"

    rand_idx = np.random.choice(embeddings.shape[0], 2, replace=False)

    style_1 = embeddings[rand_idx[0]].tolist()
    style_2 = embeddings[rand_idx[1]].tolist()

    infer_params = InferenceParameters()

    image_provider_1 = ImageProvider(
        style=style_1,
        prompt=prompt_1,
        inference_parameters=infer_params,)
    
    image_provider_2 = ImageProvider(
        style=style_2,
        prompt=prompt_2,
        inference_parameters=infer_params,)
    
    # check hash key
    assert image_provider_1.metadata != image_provider_2.metadata

    # check requesting same image
    image_provider_1a = ImageProvider(
        style=style_1,
        prompt=prompt_1,
        inference_parameters=infer_params,)
    
    assert image_provider_1.metadata == image_provider_1a.metadata
    assert image_provider_1.file_key == image_provider_1a.file_key

    # check metadata from s3
    metadata_1 = image_provider_1.get_s3_metadata_url()
    metadata_1a = image_provider_1a.get_s3_metadata_url()

    assert metadata_1 == metadata_1a

    # # compare image pillow 
    # image_1 = PILImage.open(image_provider_1.await_completion().url)
    # image_1a = PILImage.open(image_provider_1a.await_completion().url)
    # image_2 = PILImage.open(image_provider_2.await_completion().url)

    # assert np.array_equal(image_1, image_1a)
    # assert not np.array_equal(image_1, image_2)


