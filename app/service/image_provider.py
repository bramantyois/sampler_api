import uuid

from typing import List


def get_image(
    style_vector: List[float], 
    prompt: str, 
    seed: int)->str:
    # generate uuid
    file_name = uuid.uuid4()

    return str(file_name) + ".jpg"
    