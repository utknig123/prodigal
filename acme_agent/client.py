"""Candidate DB client. PATCH sends only the fields the caller names."""

from __future__ import annotations

import json
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from acme_agent.parse import values_equal
from acme_agent.schema_rules import Schema, parse_schema

DEFAULT_BASE = "https://candidate-db-api.service.external.usea2.aws.prodigaltech.com"


class ApiError(Exception):
    def __init__(self, status: int, message: str, body: object = None):
        super().__init__(message)
        self.status = status
        self.body = body


class Conflict(ApiError):
    pass


@dataclass
class Transcript:
    candidate_id: str
    path: str | None = None
    events: list[dict] = field(default_factory=list)
    source: str = "live"

    def add(self, event: dict) -> None:
        self.events.append(event)
        self.flush()

    def flush(self) -> None:
        if not self.path:
            return
        payload = {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "events": self.events,
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


class UrllibTransport:
    def __init__(self, base_url: str, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def send(self, method: str, path: str, body: dict | None, headers: dict[str, str]) -> tuple[int, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            status = exc.code
        if not raw:
            return status, None
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, {"detail": raw[:500]}


class CandidateClient:
    def __init__(self, base_url: str, api_key: str, transport=None, transcript: Transcript | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport or UrllibTransport(self.base_url)
        self.transcript = transcript
        self._schema: Schema | None = None

    def get_schema(self) -> Schema:
        if self._schema is None:
            payload = self._request("GET", "/v1/schema", None, auth=False)
            if not isinstance(payload, dict):
                raise ApiError(200, "Schema response was not an object", payload)
            self._schema = parse_schema(payload)
        return self._schema

    def list_candidates(self, collection_status: str | None = None) -> list[dict]:
        path = "/v1/candidates"
        if collection_status:
            path += f"?collection_status={urllib.request.quote(collection_status)}"
        payload = self._request("GET", path, None, auth=True)
        return unwrap_list(payload)

    def health(self) -> object:
        return self._request("GET", "/health", None, auth=False)

    def reset(self) -> object:
        return self._request("POST", "/v1/reset", None, auth=True)

    def get_candidate(self, candidate_id: str) -> dict:
        payload = self._request("GET", f"/v1/candidates/{urllib.request.quote(candidate_id)}", None, auth=True)
        return unwrap_candidate(payload)

    def patch(self, candidate_id: str, version: object, fields: dict, idempotency_key: str | None = None) -> dict:
        if not fields:
            return self.get_candidate(candidate_id)
        key = idempotency_key or str(uuid.uuid4())
        headers = {"If-Match": f'"{version}"', "Idempotency-Key": key}
        self._request(
            "PATCH",
            f"/v1/candidates/{urllib.request.quote(candidate_id)}",
            fields,
            auth=True,
            extra_headers=headers,
            idempotency_key=key,
        )
        # Trust a fresh read. A PATCH body is not guaranteed to be the whole record.
        return self.get_candidate(candidate_id)

    def patch_retry(self, record: dict, fields: dict) -> dict:
        """Re-read on 409 and send only the values that are still different."""
        current = record
        pending = dict(fields)
        schema = self.get_schema()
        last: Exception | None = None
        for _ in range(3):
            pending = {
                key: value
                for key, value in pending.items()
                if not _already(schema, key, current.get(key), value)
            }
            if not pending:
                return current
            try:
                return self.patch(str(current["id"]), current["version"], pending)
            except Conflict as exc:
                last = exc
                current = self.get_candidate(str(current["id"]))
        if last:
            raise last
        return current

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None,
        auth: bool,
        extra_headers: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> object:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            if not self.api_key:
                raise ApiError(401, "CANDIDATE_API_KEY is empty")
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)
        delays = (0, 1, 2)
        last_status = 0
        last_body: object = None
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            status, payload = self.transport.send(method, path, body, headers)
            self._log(method, path, status, body, payload, extra_headers, idempotency_key)
            if status in {200, 201}:
                return payload
            last_status, last_body = status, payload
            if status == 409:
                raise Conflict(status, _detail(payload), payload)
            if status == 401:
                raise ApiError(status, "The candidate API rejected the key.", payload)
            if status == 404:
                raise ApiError(status, "Not found.", payload)
            if status in {429, 500, 502, 503, 504} and attempt < len(delays) - 1:
                continue
            raise ApiError(status, _detail(payload), payload)
        raise ApiError(last_status, _detail(last_body), last_body)

    def _log(self, method, path, status, body, payload, extra_headers, idempotency_key) -> None:
        if not self.transcript:
            return
        safe_headers = {}
        if extra_headers:
            safe_headers = {key: value for key, value in extra_headers.items() if key.lower() != "authorization"}
        if idempotency_key:
            safe_headers["Idempotency-Key"] = idempotency_key
        self.transcript.add(
            {
                "type": "api",
                "method": method,
                "path": path,
                "status": status,
                "request": body,
                "response": payload,
                "headers": safe_headers,
            }
        )


def unwrap_candidate(payload: object) -> dict:
    if isinstance(payload, dict) and "id" in payload:
        return payload
    if isinstance(payload, dict):
        for key in ("candidate", "data", "result"):
            value = payload.get(key)
            if isinstance(value, dict) and "id" in value:
                return value
    raise ApiError(200, "Candidate response did not include an id", payload)


def unwrap_list(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("candidates", "items", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ApiError(200, "Candidate list response was not a list", payload)


def _detail(payload: object) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        return str(detail)[:500]
    return str(payload)[:500]


def _already(schema: Schema, key: str, current: object, wanted: object) -> bool:
    if key not in schema.fields:
        return current == wanted
    return values_equal(schema.spec(key), current, wanted)
