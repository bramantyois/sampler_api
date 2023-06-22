from pydantic import BaseModel
from typing import List, Union

class Request(BaseModel):
    request_id: str
    weights: Union[None, List[float]]
    prompt: str
    seed: int = 42
    n_dim_to_keep: int = 5
    std: float = 0.1
