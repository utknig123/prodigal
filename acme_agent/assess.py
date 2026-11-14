"""Decide what is missing or unacceptable on a candidate record."""

from __future__ import annotations

from dataclasses import dataclass, field

from acme_agent.parse import stored_error
from acme_agent.schema_rules import READ_ONLY, Schema


@dataclass
class Assessment:
    missing: list[str] = field(default_factory=list)
    invalid: list[tuple[str, str]] = field(default_factory=list)
    optional_invalid: list[tuple[str, str]] = field(default_factory=list)
    unwritable: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[str]:
        names = list(self.missing)
        names.extend(name for name, _reason in self.invalid)
        names.extend(name for name, _reason in self.optional_invalid)
        seen: list[str] = []
        for name in names:
            if name not in seen:
                seen.append(name)
        return seen

    @property
    def ready(self) -> bool:
        return not self.blocking and not self.unwritable


def assess(record: dict, schema: Schema) -> Assessment:
    result = Assessment()
    for name in schema.mandatory:
        spec = schema.spec(name)
        if not spec.writable:
            if _blank(record.get(name)):
                result.unwritable.append(name)
            continue
        error = stored_error(spec, record.get(name))
        if error == "missing":
            result.missing.append(name)
        elif error:
            result.invalid.append((name, error))
    for name in schema.profile_fields:
        spec = schema.spec(name)
        if spec.mandatory or name in result.missing:
            continue
        error = stored_error(spec, record.get(name))
        if error and error != "missing":
            result.optional_invalid.append((name, error))
    return result


def public_view(record: dict, schema: Schema) -> dict:
    """Fields the model is allowed to see. Id and phone stay out of the prompt."""
    hidden = set(READ_ONLY)
    view = {}
    for name in list(schema.profile_fields) + ["collection_status", "attempt_count", "notes"]:
        if name in hidden:
            continue
        value = record.get(name)
        if value not in (None, "", []):
            view[name] = value
    return view


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False
