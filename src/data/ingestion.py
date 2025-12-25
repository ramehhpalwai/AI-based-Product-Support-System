import json
from pathlib import Path
from typing import List, Tuple, Union

from pydantic import ValidationError

from src.schemas.ticket import Ticket


def load_tickets(path: Union[str, Path]) -> Tuple[List[Ticket], List[dict]]:
    """
    Returns (valid_tickets, invalid_records).

    Supports:
    - JSON array: [ {...}, {...} ]
    - NDJSON: {...}\n{...}\n
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], []

    # try JSON array / object
    try:
        obj = json.loads(text)
        records = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        # NDJSON fallback
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    valid: List[Ticket] = []
    invalid: List[dict] = []

    for rec in records:
        try:
            valid.append(Ticket.model_validate(rec))
        except ValidationError:
            invalid.append(rec)

    return valid, invalid


if __name__ == "__main__":
    valid, invalid = load_tickets("data/raw/support_tickets.json")
    print("valid:", len(valid), "invalid:", len(invalid))
    if invalid:
        print("example invalid keys:", list(invalid[0].keys())[:10])
