from pydantic import BaseModel
from typing import List


class Response(BaseModel):
    request_id: str
    weights: List[float]
    style_vector: List[float]
    url: str