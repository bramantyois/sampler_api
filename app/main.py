import os

import logging

from fastapi import FastAPI
from dotenv import load_dotenv

import numpy as np
import datasets

from models.request import Request
from models.response import Response
from sampler.sampler import sample
from service.image_provider import get_image

# create logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

app = FastAPI()

# load datasets
load_dotenv()
hf_dataset = os.getenv("HF_DATASET")
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

    image_url = get_image(
        style_vector=style_vector,
        prompt=request.prompt,
        seed=request.seed,)
    
    return Response(
        request_id=request.request_id,
        vector=style_vector,
        url=image_url,)
