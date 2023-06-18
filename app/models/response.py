from pydantic import BaseModel
from typing import List, Union

class Response(BaseModel):
    request_id: str
    vector: List[float]
    url: str