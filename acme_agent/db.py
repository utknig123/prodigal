"""Candidate DB client: retries, Retry-After, If-Match, Idempotency-Key, typed errors, call log."""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
import uuid


class DBError(Exception):
    def __init__(self, status, detail, kind="error"):
        super().__init__(f"{kind} ({status}): {detail}")
        self.status, self.detail, self.kind = status, detail, kind


class NotFound(DBError): pass
class Conflict(DBError): pass
class Rejected(DBError): pass      # 422
class AuthError(DBError): pass
class Unavailable(DBError): pass   # network / 5xx / 429 after retries


class CandidateDB:
    def __init__(self, base_url, api_key, timeout=20, retries=4, sleep=time.sleep, on_call=None):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.timeout, self.retries, self.sleep = timeout, retries, sleep
        self.on_call = on_call  # callback(entry) for the audit trace

    # ------------------------------------------------------------ public API
    def schema(self):
        return self._req("GET", "/v1/schema", auth=False)[0]

    def list(self, status=None):
        path = "/v1/candidates" + (f"?collection_status={status}" if status else "")
        data = self._req("GET", path)[0]
        if isinstance(data, dict):
            data = data.get("candidates") or data.get("items") or data.get("data") or []
        return data

    def get(self, cid):
        data = self._req("GET", f"/v1/candidates/{cid}")[0]
        return data.get("candidate", data) if isinstance(data, dict) else data

    def patch(self, cid, fields, version=None, idem_key=None):
        headers = {"Idempotency-Key": idem_key or str(uuid.uuid4())}
        if version is not None:
            headers["If-Match"] = f'"{version}"'
        data = self._req("PATCH", f"/v1/candidates/{cid}", body=fields, headers=headers)[0]
        return data.get("candidate", data) if isinstance(data, dict) else data

    def reset(self):
        return self._req("POST", "/v1/reset")[0]

    # ------------------------------------------------------------ plumbing
    def _req(self, method, path, body=None, headers=None, auth=True):
        url = self.base + path
        hdrs = {"Accept": "application/json", "User-Agent": "acme-intake-agent/1.0"}
        if auth:
            hdrs["Authorization"] = f"Bearer {self.key}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        last = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            status, payload, retry_after = None, None, None
            try:
                req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    status, raw = r.status, r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                status, raw = e.code, e.read().decode("utf-8", "replace")
                retry_after = e.headers.get("Retry-After")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                status, raw = 0, str(e)
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                payload = {"detail": raw[:300]}
            self._log(method, path, body, status, payload, int((time.time() - t0) * 1000), attempt)
            if 200 <= status < 300:
                return payload, status
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            if status == 404:
                raise NotFound(status, detail, "not_found")
            if status == 409:
                raise Conflict(status, detail, "conflict")
            if status == 422:
                raise Rejected(status, detail, "rejected")
            if status in (401, 403):
                raise AuthError(status, detail, "auth")
            if status in (0, 429) or status >= 500:
                last = DBError(status, detail, "unavailable")
                if attempt < self.retries:
                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = min(8, 0.6 * 2 ** attempt) + random.random() * 0.3
                    self.sleep(wait)
                    continue
                raise Unavailable(status, detail, "unavailable")
            raise DBError(status, detail)
        raise last or Unavailable(0, "unknown")

    def _log(self, method, path, body, status, payload, ms, attempt):
        if self.on_call:
            short = payload
            if isinstance(payload, dict) and len(json.dumps(payload)) > 1500:
                short = {k: payload[k] for k in list(payload)[:8]}
            self.on_call({"method": method, "path": path, "request": body, "status": status,
                          "response": short, "ms": ms, "attempt": attempt})
