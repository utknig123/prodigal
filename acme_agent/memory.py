"""In-memory candidate API used by tests and the labeled sample runs."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

from acme_agent.schema_rules import READ_ONLY, load_snapshot

WRITABLE_BLOCKED = set(READ_ONLY)


def seed_record(candidate_id: str = "c001", **overrides) -> dict:
    """A valid record. Tests override the fields they want to exercise."""
    record = {
        "id": candidate_id,
        "full_name": "Meera Iyer",
        "phone": "+919800000001",
        "email": "meera@example.com",
        "college": "NMIMS Mumbai",
        "degree": "B.Tech",
        "branch": "Computer Engineering",
        "graduation_year": 2027,
        "cgpa": 8.4,
        "preferred_location": "Mumbai",
        "available_from": "2026-06-01",
        "internship_duration_months": 6,
        "linkedin_url": None,
        "github_url": None,
        "skills": None,
        "collection_status": "pending",
        "attempt_count": 0,
        "last_contacted_at": None,
        "notes": "",
        "version": 1,
        "updated_at": "2026-11-01T00:00:00+00:00",
    }
    record.update(overrides)
    return record


class MemoryTransport:
    def __init__(self, records: list[dict]):
        self.records = {record["id"]: deepcopy(record) for record in records}
        self.schema = load_snapshot()
        self.idempotency: dict[str, tuple[str, int, object]] = {}

    def send(self, method: str, path: str, body: dict | None, headers: dict[str, str]) -> tuple[int, object]:
        if method == "GET" and path.split("?", 1)[0] == "/v1/schema":
            return 200, deepcopy(self.schema)
        if method == "GET" and path.split("?", 1)[0] == "/v1/candidates":
            return 200, self._list(path)
        if path.startswith("/v1/candidates/"):
            candidate_id = path.rsplit("/", 1)[-1]
            if method == "GET":
                record = self.records.get(candidate_id)
                if record is None:
                    return 404, {"detail": "not found"}
                return 200, deepcopy(record)
            if method == "PATCH":
                return self._patch(candidate_id, body or {}, headers)
        return 404, {"detail": f"no route {method} {path}"}

    def _list(self, path: str) -> list[dict]:
        status = ""
        if "?" in path and "collection_status=" in path:
            status = path.split("collection_status=", 1)[1].split("&", 1)[0]
        rows = [deepcopy(record) for record in self.records.values()]
        if status:
            rows = [row for row in rows if row.get("collection_status") == status]
        return rows

    def _patch(self, candidate_id: str, body: dict, headers: dict[str, str]) -> tuple[int, object]:
        record = self.records.get(candidate_id)
        if record is None:
            return 404, {"detail": "not found"}
        key = headers.get("Idempotency-Key", "")
        signature = json.dumps(body, sort_keys=True)
        if key and key in self.idempotency:
            previous, status, payload = self.idempotency[key]
            if previous == signature:
                return status, deepcopy(payload)
            return 409, {"detail": "idempotency key reused with a different body"}
        unknown = [name for name in body if name not in self._known()]
        readonly = [name for name in body if name in WRITABLE_BLOCKED]
        if unknown or readonly:
            return 422, {"detail": {"unknown": unknown, "read_only": readonly}}
        expected = str(headers.get("If-Match", "")).strip().strip('"')
        if expected != str(record["version"]):
            return 409, {"detail": "stale version", "version": record["version"]}
        for name, value in body.items():
            record[name] = value
        record["version"] = int(record["version"]) + 1
        record["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        payload = deepcopy(record)
        if key:
            self.idempotency[key] = (signature, 200, payload)
        return 200, payload

    def _known(self) -> set[str]:
        return {field["name"] for field in self.schema["fields"]}
