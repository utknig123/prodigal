"""HTTP layer shared by local server and AWS Lambda. Stateless: the signed chat state travels with the client."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import zlib
from pathlib import Path

from . import validators as V
from .agent import Conversation
from .config import ROOT, load_env, make_clients
from .db import AuthError, DBError, NotFound

FRONTEND = (ROOT / "frontend").resolve()
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".svg": "image/svg+xml", ".json": "application/json"}
_clients = {}


def clients():
    if "c" not in _clients:
        _clients["c"] = make_clients()
    return _clients["c"]


def _secret():
    return (os.environ.get("SESSION_SECRET") or os.environ.get("LLM_API_KEY") or "dev-secret").encode()


def pack(state):
    raw = zlib.compress(json.dumps(state, default=str).encode(), 6)
    body = base64.urlsafe_b64encode(raw).decode()
    return body + "." + hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()[:32]


def unpack(token):
    body, _, sig = (token or "").rpartition(".")
    if not hmac.compare_digest(sig, hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()[:32]):
        raise ValueError("invalid session token")
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(body)))


def view(conv, extra=None):
    s = conv.s
    audit = V.audit_record(s["record"], set(s.get("verified") or []))
    plan = audit.to_dict()
    focus = (audit.todo_fields() or audit.optional_missing or [""])[0]
    return {"token": pack(s), "transcript": s["transcript"], "trace": s["trace"][-80:], "plan": plan,
            "record": s["record"], "initial": s["initial"], "saved": s["saved"], "ended": s["ended"],
            "outcome": s["outcome"], "identity": s["identity"], "candidate_id": s["cid"],
            "focus": focus, "handoff": s.get("handoff"),
            "final_write_ok": s.get("final_write_ok", True), **(extra or {})}


def dispatch(method, path, body, headers=None):
    method = (method or "GET").upper()
    route = (path or "/").split("?", 1)[0]
    route = route.rstrip("/") or "/"
    if method == "OPTIONS":
        return 204, _h("text/plain"), ""
    try:
        if route.startswith("/api/"):
            return _api(method, route, body)
    except NotFound:
        return _j(404, {"error": "Candidate not found. Check the ID (e.g. c001)."})
    except AuthError:
        return _j(502, {"error": "Candidate API rejected the API key. Check CANDIDATE_API_KEY."})
    except DBError as e:
        return _j(502, {"error": f"Candidate database problem: {e.detail}"})
    except ValueError as e:
        return _j(400, {"error": str(e)})
    except Exception as e:  # last-resort: never leak a stack trace
        return _j(500, {"error": f"Unexpected error: {type(e).__name__}"})
    if method in ("GET", "HEAD"):
        return _static(route)
    return _j(405, {"error": "Method not allowed"})


def _api(method, route, body):
    db, llm = clients()
    data = {}
    if method == "POST":
        try:
            data = json.loads(body or "{}")
        except ValueError:
            raise ValueError("Body must be JSON")
    if route == "/api/health" and method == "GET":
        chain = [{"provider": p, "model": m} for p, _b, _k, m in getattr(llm, "targets", [])]
        out = {"ok": True, "llm": {"provider": llm.provider, "model": llm.model, "configured": llm.configured, "chain": chain}}
        try:
            db.schema()
            out["candidate_api_schema"] = "ok"
            out["candidate_api_auth"] = "ok" if db.key and db.list() is not None else "missing key"
        except Exception as e:
            out["candidate_api_auth"] = f"error: {str(e)[:120]}"
        return _j(200, out)
    if route == "/api/candidates" and method == "GET":
        rows = [{"id": c["id"], "name": c.get("full_name"), "status": c.get("collection_status"),
                 "attempts": c.get("attempt_count"), "todo": V.audit_record(c).todo_fields()} for c in db.list()]
        return _j(200, {"candidates": rows})
    if route == "/api/reset" and method == "POST":
        db.reset()
        return _j(200, {"ok": True})
    if route == "/api/start" and method == "POST":
        cid = str(data.get("candidate_id", "")).strip()
        if not cid:
            raise ValueError("candidate_id required")
        if not llm.configured:
            return _j(503, {"error": "LLM_API_KEY is not configured on the server."})
        conv, audit = Conversation.start(cid, db, llm)
        if conv.s["ended"]:
            return _j(200, view(conv, {"no_action": True}))
        conv.open()
        return _j(200, view(conv))
    if route in ("/api/message", "/api/leave") and method == "POST":
        conv = Conversation(unpack(data.get("token")), db, llm)
        if route == "/api/leave":
            conv.abandon(str(data.get("reason") or "candidate left the chat"))
        elif not conv.s["ended"]:
            text = str(data.get("text", "")).strip()
            if not text:
                raise ValueError("Message is empty")
            conv.turn(text)
        return _j(200, view(conv))
    return _j(404, {"error": "Not found"})


def _static(route):
    rel = "index.html" if route == "/" else route.lstrip("/")
    p = (FRONTEND / rel).resolve()
    if FRONTEND not in p.parents or not p.is_file():
        return _j(404, {"error": "Not found"})
    return 200, _h(TYPES.get(p.suffix, "application/octet-stream")), p.read_text(encoding="utf-8")


def _j(code, obj):
    return code, _h("application/json; charset=utf-8"), json.dumps(obj, default=str)


def _h(ct):
    return {
        "content-type": ct,
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "content-type",
        "access-control-allow-methods": "GET, POST, OPTIONS",
    }
