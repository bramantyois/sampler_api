from pydantic import BaseModel
from typing import List, Union

class Request(BaseModel):
    request_id: str
    weights: Union[None, List[float]]
    prompt: str
    seed: int
