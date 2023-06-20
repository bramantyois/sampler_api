import os

import logging

from fastapi import FastAPI
from dotenv import load_dotenv

import numpy as np
import datasets

from models.request import Request
from models.response import Response
from sampler.sampler import sample
from models.image import Image, InferenceParameters, ImageProvider, GenerateImagesRequest, GeneratedImagesResponse

import config  
# create logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

app = FastAPI()

# load datasets
load_dotenv('hf.env')
load_dotenv('aws.env')
load_dotenv('backend.env')
        

hf_dataset = config.HF_DATASET
hf_dataset = datasets.load_dataset(hf_dataset)

embeddings = hf_dataset["train"]["embedding"]
embeddings = np.vstack(embeddings)

@app.post("/get_image")
def mutate(request: Request):
    weights = request.weights

    # get new weights
    weights = sample(
        n_hist_artists=embeddings.shape[0],
        weights=weights,)
    
    # matrix multiply embeddings with weights
    style_vector = weights @ embeddings
    style_vector = style_vector.flatten().tolist()

    infer_params = InferenceParameters()
    
    image_provider = ImageProvider(
        style=style_vector,
        prompt=request.prompt,
        inference_parameters=infer_params,)

    return Response(
        request_id=request.request_id,
        vector=style_vector,
        url=image_provider.get_s3_image_url,)
