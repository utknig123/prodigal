"""Turn GET /v1/schema into field rules the harness enforces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).with_name("schema_snapshot.json")

COLLECTION_FIELDS = (
    "collection_status",
    "attempt_count",
    "last_contacted_at",
    "notes",
)
READ_ONLY = ("id", "phone", "version", "updated_at")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type_name: str
    mandatory: bool
    writable: bool
    rules: str
    choices: tuple[str, ...] = ()
    min_len: int | None = None
    max_len: int | None = None
    min_num: float | None = None
    max_num: float | None = None
    number_is_int: bool = False
    max_items: int | None = None
    item_min: int | None = None
    item_max: int | None = None
    min_date: str | None = None
    max_date: str | None = None

    @property
    def is_enum(self) -> bool:
        return bool(self.choices) and "date" not in self.type_name


@dataclass(frozen=True)
class Schema:
    fields: dict[str, FieldSpec]
    mandatory: tuple[str, ...]
    statuses: tuple[str, ...]
    validation_note: str

    def spec(self, name: str) -> FieldSpec:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(f"Unknown field {name!r}") from exc

    @property
    def profile_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, spec in self.fields.items()
            if spec.writable and name not in COLLECTION_FIELDS
        )


def load_snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def parse_schema(payload: dict) -> Schema:
    fields: dict[str, FieldSpec] = {}
    for raw in payload["fields"]:
        fields[raw["name"]] = _field(raw)
    mandatory = tuple(payload["mandatory_fields"])
    for name in mandatory:
        if name not in fields:
            raise ValueError(f"Schema lists unknown mandatory field {name}")
    return Schema(
        fields=fields,
        mandatory=mandatory,
        statuses=tuple(payload["collection_statuses"]),
        validation_note=str(payload.get("validation") or ""),
    )


def _field(raw: dict) -> FieldSpec:
    rules = str(raw.get("rules") or "")
    type_name = str(raw.get("type") or "")
    choices = _choices(rules)
    min_len, max_len = _char_bounds(rules)
    min_num, max_num = _numeric_bounds(rules, type_name)
    max_items, item_min, item_max = _item_bounds(rules)
    min_date, max_date = _date_bounds(rules)
    number_is_int = "integer" in type_name or (
        bool(choices) and all(_is_int_token(item) for item in choices)
    )
    if number_is_int and choices:
        # "2, 3, 6" is an enum of ints, not a 2-6 range.
        min_num, max_num = None, None
    return FieldSpec(
        name=raw["name"],
        type_name=type_name,
        mandatory=bool(raw.get("mandatory")),
        writable=bool(raw.get("writable")),
        rules=rules,
        choices=choices,
        min_len=min_len,
        max_len=max_len,
        min_num=min_num,
        max_num=max_num,
        number_is_int=number_is_int and not choices,
        max_items=max_items,
        item_min=item_min,
        item_max=item_max,
        min_date=min_date,
        max_date=max_date,
    )


def _choices(rules: str) -> tuple[str, ...]:
    match = re.search(r"One of:\s*(.+)", rules)
    if not match:
        return ()
    raw = match.group(1).strip().rstrip(".")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _char_bounds(rules: str) -> tuple[int | None, int | None]:
    pair = re.search(r"(\d+)\s*-\s*(\d+)\s+chars", rules)
    if pair:
        return int(pair.group(1)), int(pair.group(2))
    cap = re.search(r"max\s+(\d+)\s+chars", rules)
    if cap:
        return None, int(cap.group(1))
    return None, None


def _item_bounds(rules: str) -> tuple[int | None, int | None, int | None]:
    match = re.search(r"Up to (\d+) items, each (\d+)-(\d+) chars", rules)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _date_bounds(rules: str) -> tuple[str | None, str | None]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", rules)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _numeric_bounds(rules: str, type_name: str) -> tuple[float | None, float | None]:
    if not any(token in type_name for token in ("integer", "number")):
        return None, None
    if "date" in type_name or "One of:" in rules:
        return None, None
    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", rules)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _is_int_token(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", value))
