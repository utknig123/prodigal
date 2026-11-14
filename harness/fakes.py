"""In-memory candidate DB (same semantics as the real API) + scripted LLM, for tests/harness."""
import copy

from acme_agent import db as D
from acme_agent.llm import Reply

BASE = dict(attempt_count=0, last_contacted_at=None, notes=None, linkedin_url=None, github_url=None, skills=None,
            collection_status="pending", version=1)
FULL = dict(email="x@gmail.com", college="IIT Bombay", degree="B.Tech", branch="Computer Engineering",
            graduation_year=2027, cgpa=8.4, preferred_location="Mumbai", available_from="2027-06-01",
            internship_duration_months=3)


def seed():
    def mk(i, name, **kw):
        row = {**BASE, "id": i, "full_name": name, "phone": f"+9198000000{i[-2:]}", "updated_at": "t", **{k: None for k in FULL}}
        row.update(kw)
        return row
    return {r["id"]: r for r in [
        mk("c001", "Aarav Sharma", email="aarav@example.com", college="VJTI Mumbai", degree="B.Tech"),
        mk("c002", "Priya Nair", **FULL),                                              # already complete
        mk("c003", "Rohan Mehta", **{**FULL, "cgpa": 85, "degree": "Engineering"}),    # dirty data
        mk("c004", "Sneha Iyer", **{**FULL, "email": "sneha@gmial.com", "branch": None}),
        mk("c005", "Karan Singh", **{**FULL, "degree": "btech", "cgpa": "8.5"}),       # auto-normalisable
        mk("c006", "Isha Verma", collection_status="in_progress", notes="asked to continue after 6pm", email="isha@x.com"),
        mk("c007", "Dev Patel", **{**FULL, "college": "N/A"}),
        mk("c008", "Meera Joshi", **{**FULL, "attempt_count": 49, "branch": None}),
    ]}


class FakeDB(D.CandidateDB):
    def __init__(self, fail_next=0, conflict_next=0):
        super().__init__("http://fake", "k", sleep=lambda s: None)
        self.rows, self.writes, self.idem = seed(), [], {}
        self.fail_next, self.conflict_next = fail_next, conflict_next

    def schema(self):
        return {}

    def list(self, status=None):
        return [copy.deepcopy(r) for r in self.rows.values()]

    def reset(self):
        self.rows = seed()
        return {"ok": True}

    def get(self, cid):
        if cid not in self.rows:
            raise D.NotFound(404, "not found", "not_found")
        self._log("GET", f"/v1/candidates/{cid}", None, 200, {"version": self.rows[cid]["version"]}, 1, 0)
        return copy.deepcopy(self.rows[cid])

    def patch(self, cid, fields, version=None, idem_key=None):
        if self.fail_next:
            self.fail_next -= 1
            raise D.Unavailable(503, "down", "unavailable")
        r = self.rows[cid]
        if self.conflict_next:
            self.conflict_next -= 1
            r["version"] += 1
            r["email"] = r["email"] or "other@writer.com"      # a concurrent writer
            raise D.Conflict(409, "stale", "conflict")
        if not fields or any(k in ("id", "phone", "version", "updated_at") for k in fields):
            raise D.Rejected(422, "bad", "rejected")
        if version is not None and int(version) != r["version"]:
            raise D.Conflict(409, "stale", "conflict")
        if idem_key in self.idem:
            return copy.deepcopy(self.idem[idem_key])
        self.writes.append({"id": cid, "fields": copy.deepcopy(fields), "old": {k: r.get(k) for k in fields}})
        r.update(copy.deepcopy(fields))
        r["version"] += 1
        self._log("PATCH", f"/v1/candidates/{cid}", fields, 200, {"version": r["version"]}, 1, 0)
        self.idem[idem_key] = copy.deepcopy(r)
        return copy.deepcopy(r)


class ScriptedLLM:
    """Plays back canned replies: each item is text, (tool_name, args) or a list of those."""
    configured = True

    def __init__(self, script):
        self.script, self.i = list(script), 0

    def chat(self, messages, tools=None, **kw):
        item = self.script[self.i] if self.i < len(self.script) else "ok"
        self.i += 1
        if isinstance(item, str):
            return Reply(item, [], "scripted")
        calls = item if isinstance(item, list) else [item]
        return Reply("", [{"id": f"c{self.i}{j}", "name": n, "arguments": a} for j, (n, a) in enumerate(calls)], "scripted")
