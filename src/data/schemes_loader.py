from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

_DATA_FILE = Path(__file__).with_name("schemes_data.json")

with _DATA_FILE.open("r", encoding="utf-8") as fp:
    _raw = json.load(fp)

SCHEME_KEYWORDS: Dict[str, list[str]] = _raw["SCHEME_KEYWORDS"]
SCHEME_DATA: Dict[str, Dict[str, str]] = _raw["SCHEME_DATA"]
