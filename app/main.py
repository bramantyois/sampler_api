import os

import logging

from fastapi import FastAPI
from dotenv import load_dotenv

import numpy as np
import datasets

from app.models.request import Request
from app.models.response import Response
from app.sampler.sampler import sample
from app.models.image import InferenceParameters, ImageProvider


app = FastAPI()

load_dotenv()
        
hf_dataset = os.getenv("HF_DATASET")
hf_dataset = datasets.load_dataset(hf_dataset)

embeddings = hf_dataset["train"]["embedding"]
embeddings = np.vstack(embeddings)

@app.post("/get_image")
def mutate(request: Request):
    weights = request.weights
    n_dim_to_keep = request.n_dim_to_keep
    std = request.std

    # get new weights
    weights = sample(
        n_hist_artists=embeddings.shape[0],
        weights=weights,
        n_dim_to_keep=n_dim_to_keep,
        std=std,)
    
    # matrix multiply embeddings with weights
    style_vector = weights @ embeddings
    style_vector = style_vector.flatten()

    infer_params = InferenceParameters()
    
    image_provider = ImageProvider(
        style=style_vector.tolist(),
        prompt=request.prompt,
        inference_parameters=infer_params,)

    return Response(
        request_id=request.request_id,
        style_vector=style_vector.tolist(),
        weights=weights.flatten().tolist(),
        url=image_provider.get_s3_image_url(),)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)