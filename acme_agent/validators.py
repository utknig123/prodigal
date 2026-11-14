"""Deterministic validation / normalisation of every writable field + record audit.

The candidate DB does not validate anything, so this module is the single source of truth
for what is allowed to reach the database.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import urlparse

MANDATORY = [
    "full_name", "phone", "email", "college", "degree", "branch", "graduation_year",
    "cgpa", "preferred_location", "available_from", "internship_duration_months",
]
ASKABLE = [f for f in MANDATORY if f not in ("full_name", "phone")]
OPTIONAL = ["linkedin_url", "github_url", "skills"]
WRITABLE = ASKABLE + OPTIONAL + ["full_name"]
DEGREES = ["B.Tech", "B.E.", "B.Sc", "BCA", "MBA Tech", "MCA", "M.Tech", "Other"]
LOCATIONS = ["Mumbai", "Bengaluru", "Remote", "No preference"]
DURATIONS = [2, 3, 6]
STATUSES = ["pending", "in_progress", "completed", "no_answer", "refused", "needs_human"]
PLACEHOLDERS = {"", "n/a", "na", "none", "null", "nil", "-", "--", "unknown", "tbd", "?", "undefined", "nan", "not provided"}

_DEGREE_ALIASES = {
    "btech": "B.Tech", "bacheloroftechnology": "B.Tech", "bachelorsoftechnology": "B.Tech",
    "be": "B.E.", "bachelorofengineering": "B.E.", "bachelorsofengineering": "B.E.", "beng": "B.E.",
    "bsc": "B.Sc", "bachelorofscience": "B.Sc", "bachelorsofscience": "B.Sc", "bs": "B.Sc",
    "bca": "BCA", "bachelorofcomputerapplications": "BCA",
    "mbatech": "MBA Tech", "mbatechnology": "MBA Tech",
    "mca": "MCA", "masterofcomputerapplications": "MCA", "mastersofcomputerapplications": "MCA",
    "mtech": "M.Tech", "masteroftechnology": "M.Tech", "mastersoftechnology": "M.Tech",
    "other": "Other",
}
_LOC_ALIASES = {
    "mumbai": "Mumbai", "bombay": "Mumbai", "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "blr": "Bengaluru", "bengalooru": "Bengaluru", "remote": "Remote", "wfh": "Remote",
    "workfromhome": "Remote", "nopreference": "No preference", "nopref": "No preference",
    "any": "No preference", "anywhere": "No preference", "flexible": "No preference",
    "anylocation": "No preference", "doesntmatter": "No preference",
}
_NUM_WORDS = {"two": 2, "three": 3, "six": 6, "2": 2, "3": 3, "6": 6}
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
_TYPO_DOMAINS = {"gmial.com", "gmai.com", "gamil.com", "gmail.con", "gmail.co", "gmal.com", "gmaill.com",
                 "yaho.com", "yahooo.com", "yahoo.con", "hotmial.com", "hotmail.con", "outlok.com",
                 "outlook.con", "gnail.com", "gmail.in", "gmail.cm"}
_FAKE_EMAILS = {"test@test.com", "abc@abc.com", "a@a.com", "example@example.com", "xyz@xyz.com", "email@email.com"}


@dataclass
class Result:
    ok: bool
    value: object = None
    error: str = ""
    confirm: str = ""          # valid but suspicious: ask the candidate to confirm first
    notes: list = field(default_factory=list)


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in PLACEHOLDERS
    if isinstance(v, (list, tuple)):
        return len(v) == 0
    return False


def _bad(msg):
    return Result(False, error=msg)


def validate(name: str, value, today: dt.date | None = None) -> Result:
    today = today or dt.date.today()
    fn = _VALIDATORS.get(name)
    if fn is None:
        return _bad(f"'{name}' is not a field the agent may write.")
    if name in OPTIONAL and value is None:
        return Result(True, None)
    if name != "skills" and is_blank(value):
        return _bad("value is empty or a placeholder")
    try:
        return fn(value, today)
    except Exception as exc:  # defensive: never let a validator crash the agent
        return _bad(f"could not parse value ({exc})")


def _text(lo, hi, label):
    def f(v, today):
        if not isinstance(v, str):
            return _bad(f"{label} must be text")
        s = re.sub(r"\s+", " ", v).strip()
        if not (lo <= len(s) <= hi):
            return _bad(f"{label} must be {lo}-{hi} characters")
        if re.fullmatch(r"[\W\d_]+", s):
            return _bad(f"{label} does not look like real text")
        if label == "branch" and s.islower():
            s = s.title()
        return Result(True, s)
    return f


def _email(v, today):
    if not isinstance(v, str):
        return _bad("email must be text")
    s = v.strip().strip("<>").replace(" ", "")
    if len(s) > 254 or not _EMAIL_RE.match(s) or ".." in s:
        return _bad("not a valid email address")
    local, domain = s.rsplit("@", 1)
    s = f"{local}@{domain.lower()}"
    r = Result(True, s)
    if domain.lower() in _TYPO_DOMAINS or s.lower() in _FAKE_EMAILS:
        r.confirm = f"'{s}' looks like a typo or placeholder address; read it back to the candidate and confirm"
    return r


def _degree(v, today):
    k = _key(str(v))
    if k in _DEGREE_ALIASES:
        return Result(True, _DEGREE_ALIASES[k])
    return _bad("degree must be one of " + ", ".join(DEGREES) + " (use 'Other' for anything else and note the real degree)")


def _location(v, today):
    k = _key(str(v))
    if k in _LOC_ALIASES:
        return Result(True, _LOC_ALIASES[k])
    return _bad("preferred_location must be one of " + ", ".join(LOCATIONS))


def _int(v, lo, hi, label):
    if isinstance(v, bool):
        raise ValueError("boolean")
    if isinstance(v, float) and v != int(v):
        raise ValueError("not whole")
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d{4}|\d+)\s*", v)
        if not m:
            raise ValueError("not a number")
        v = int(m.group(1))
    v = int(v)
    if not lo <= v <= hi:
        return _bad(f"{label} must be between {lo} and {hi}")
    return Result(True, v)


def _grad(v, today):
    return _int(v, 2026, 2028, "graduation_year")


def _cgpa(v, today):
    if isinstance(v, bool):
        return _bad("cgpa must be a number")
    if isinstance(v, str):
        s = v.strip()
        if "%" in s:
            return _bad("that is a percentage; ask for the CGPA on the 10-point scale")
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?", s)
        if not m:
            return _bad("cgpa must be a number")
        if m.group(2) and float(m.group(2)) != 10:
            return _bad(f"cgpa is out of {m.group(2)}, not the 10-point scale; do not convert yourself, ask the candidate")
        v = float(m.group(1))
    d = Decimal(str(v))
    if d < 0 or d > 10:
        return _bad("cgpa must be 0-10 on a 10-point scale (values like 85 are probably a percentage; ask for CGPA)")
    d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    r = Result(True, float(d))
    if d <= Decimal("4.5"):
        r.confirm = f"{d} is unusually low; it may be on a 4-point scale. Confirm it is out of 10"
    return r


def _date(v, today):
    if not isinstance(v, str):
        return _bad("available_from must be a YYYY-MM-DD string")
    try:
        d = dt.date.fromisoformat(v.strip())
    except ValueError:
        return _bad("available_from must be a real date formatted YYYY-MM-DD")
    if not (dt.date(2026, 1, 1) <= d <= dt.date(2027, 12, 31)):
        return _bad("available_from must be between 2026-01-01 and 2027-12-31")
    r = Result(True, d.isoformat())
    if d < today:
        r.confirm = f"{d} is in the past; confirm the candidate really means that date"
    return r


def _duration(v, today):
    if isinstance(v, bool):
        return _bad("duration must be 2, 3 or 6 months")
    if isinstance(v, str):
        m = re.search(r"\b(\d+|two|three|six)\b", v.lower())
        v = _NUM_WORDS.get(m.group(1), m.group(1)) if m else v
    try:
        n = int(v)
    except (TypeError, ValueError):
        return _bad("duration must be 2, 3 or 6 months")
    if n not in DURATIONS:
        return _bad("internship_duration_months must be one of 2, 3, 6")
    return Result(True, n)


def _url(domain, label, profile_only=False):
    def f(v, today):
        if not isinstance(v, str):
            return _bad(f"{label} must be a URL")
        s = v.strip()
        if not re.match(r"^https?://", s, re.I):
            s = "https://" + s
        u = urlparse(s)
        host = (u.hostname or "").lower()
        if not (host == domain or host.endswith("." + domain)):
            return _bad(f"{label} must be a {domain} URL")
        parts = [p for p in u.path.split("/") if p]
        if not parts:
            return _bad(f"{label} must point to a profile, not the {domain} homepage")
        if profile_only and len(parts) != 1:
            return _bad("github_url must be a profile URL like https://github.com/username (not a repo)")
        return Result(True, f"https://{host}{u.path.rstrip('/')}")
    return f


def _skills(v, today):
    if v is None or v == "":
        return _bad("skills is empty")
    if isinstance(v, str):
        v = [x for x in re.split(r"[,;\n]", v)]
    if not isinstance(v, (list, tuple)):
        return _bad("skills must be a list of strings")
    out, seen = [], set()
    for item in v:
        s = re.sub(r"\s+", " ", str(item)).strip()
        if not s or s.lower() in seen:
            continue
        if len(s) > 50:
            return _bad(f"skill '{s[:20]}...' is longer than 50 chars")
        seen.add(s.lower())
        out.append(s)
    if not out:
        return _bad("skills is empty")
    if len(out) > 30:
        return _bad("at most 30 skills")
    return Result(True, out)


_VALIDATORS = {
    "full_name": _text(2, 100, "full_name"),
    "email": _email,
    "college": _text(2, 150, "college"),
    "degree": _degree,
    "branch": _text(2, 100, "branch"),
    "graduation_year": _grad,
    "cgpa": _cgpa,
    "preferred_location": _location,
    "available_from": _date,
    "internship_duration_months": _duration,
    "linkedin_url": _url("linkedin.com", "linkedin_url"),
    "github_url": _url("github.com", "github_url", profile_only=True),
    "skills": _skills,
}


# ---------------------------------------------------------------- record audit
@dataclass
class Audit:
    missing: list = field(default_factory=list)
    invalid: list = field(default_factory=list)       # [{field, value, reason, suggestion}]
    confirm: list = field(default_factory=list)       # [{field, value, reason}]
    autofix: dict = field(default_factory=dict)       # unambiguous canonicalisations
    ok: list = field(default_factory=list)
    optional_missing: list = field(default_factory=list)

    @property
    def needs_conversation(self) -> bool:
        return bool(self.missing or self.invalid or self.confirm)

    @property
    def complete(self) -> bool:
        return not self.needs_conversation

    def todo_fields(self) -> list:
        return list(dict.fromkeys(self.missing + [i["field"] for i in self.invalid] + [i["field"] for i in self.confirm]))

    def to_dict(self):
        return {"missing": self.missing, "invalid": self.invalid, "confirm": self.confirm,
                "autofix": self.autofix, "ok": self.ok, "optional_missing": self.optional_missing}


def audit_record(rec: dict, verified: set | None = None, today: dt.date | None = None) -> Audit:
    verified = verified or set()
    a = Audit()
    for f in MANDATORY:
        v = rec.get(f)
        if f == "phone":
            if is_blank(v):
                a.missing.append(f)
            else:
                a.ok.append(f)
            continue
        if is_blank(v):
            a.missing.append(f)
            continue
        r = validate(f, v, today)
        if not r.ok:
            a.invalid.append({"field": f, "value": v, "reason": r.error})
        elif r.confirm and f not in verified:
            a.confirm.append({"field": f, "value": v, "reason": r.confirm})
        else:
            a.ok.append(f)
            if r.value != v and f in ("degree", "preferred_location", "graduation_year", "cgpa",
                                      "internship_duration_months", "college", "branch", "email", "available_from"):
                # same information, canonical representation (type / case / spacing)
                if f in ("college",) and isinstance(v, str) and re.sub(r"\s+", " ", v).strip() == r.value:
                    a.autofix[f] = r.value
                elif f not in ("college", "branch") or str(v).strip() != r.value:
                    a.autofix[f] = r.value
    for f in OPTIONAL:
        v = rec.get(f)
        if is_blank(v):
            a.optional_missing.append(f)
            continue
        r = validate(f, v, today)
        if not r.ok:
            a.invalid.append({"field": f, "value": v, "reason": r.error})
        elif r.value != v:
            a.autofix[f] = r.value
    return a
