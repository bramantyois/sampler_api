from pydantic import BaseModel
from typing import List, Union

class Response(BaseModel):
    request_id: str
    weights: List[float]
    style_vector: List[float]
    url: str