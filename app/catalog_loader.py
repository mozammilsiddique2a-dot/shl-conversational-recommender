import json
from functools import lru_cache
from pathlib import Path
from typing import List

from app.schemas import Assessment


CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> List[Assessment]:
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)
    return [Assessment.model_validate(item) for item in raw_items]
