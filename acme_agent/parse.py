"""Parse a candidate quote into a canonical field value. No guessing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import urlparse

from acme_agent.quotes import find_span
from acme_agent.schema_rules import FieldSpec

PLACEHOLDERS = frozenset(
    {"n/a", "na", "none", "null", "nil", "test", "todo", "tbd", "unknown", "-", "--", ".", "na."}
)

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_DEGREE_ALIASES = (
    ("bachelor of technology", "B.Tech"),
    ("b.tech", "B.Tech"),
    ("b tech", "B.Tech"),
    ("btech", "B.Tech"),
    ("bachelor of engineering", "B.E."),
    ("b.e.", "B.E."),
    ("b.e", "B.E."),
    ("bachelor of science", "B.Sc"),
    ("b.sc.", "B.Sc"),
    ("b.sc", "B.Sc"),
    ("bsc", "B.Sc"),
    ("mba tech", "MBA Tech"),
    ("mbatech", "MBA Tech"),
    ("m.tech", "M.Tech"),
    ("m tech", "M.Tech"),
    ("mtech", "M.Tech"),
    ("bca", "BCA"),
    ("mca", "MCA"),
    ("other", "Other"),
)
_LOCATION_ALIASES = (
    ("no preference", "No preference"),
    ("no pref", "No preference"),
    ("anywhere", "No preference"),
    ("work from home", "Remote"),
    ("wfh", "Remote"),
    ("remote", "Remote"),
    ("bengaluru", "Bengaluru"),
    ("bangalore", "Bengaluru"),
    ("mumbai", "Mumbai"),
    ("bombay", "Mumbai"),
)
_DURATION_WORDS = {"two": 2, "three": 3, "six": 6}
_BARE_DEGREE = {"be": "B.E.", "b.e": "B.E.", "b.e.": "B.E."}
_GITHUB_RESERVED = frozenset(
    {
        "about", "features", "pricing", "login", "signup", "topics", "explore",
        "settings", "marketplace", "orgs", "organizations", "apps", "sponsors",
    }
)


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    value: object = None
    error: str = ""


def interpret(spec: FieldSpec, proposed: object, evidence: str) -> ParseResult:
    """Accept proposed only when evidence itself yields that same value."""
    if spec.name in {"email"}:
        return _match_extracted(spec, proposed, _one_email(evidence), evidence)
    if spec.name == "cgpa":
        return _match_extracted(spec, proposed, _one_cgpa(evidence), evidence)
    if spec.name == "graduation_year":
        return _match_extracted(spec, proposed, _one_year(spec, evidence), evidence)
    if spec.name == "available_from":
        return _match_extracted(spec, proposed, _one_date(spec, evidence), evidence)
    if spec.name == "degree":
        return _match_extracted(spec, proposed, _one_degree(spec, evidence), evidence)
    if spec.name == "preferred_location":
        return _match_extracted(spec, proposed, _one_location(spec, evidence), evidence)
    if spec.name == "internship_duration_months":
        return _match_extracted(spec, proposed, _one_duration(spec, evidence), evidence)
    if spec.name in {"linkedin_url", "github_url"}:
        return _match_extracted(spec, proposed, _one_url(spec, evidence), evidence)
    if spec.name == "skills":
        return _skills(spec, proposed, evidence)
    if spec.name in {"full_name", "college", "branch"}:
        return _free_text(spec, proposed, evidence)
    return ParseResult(False, error=f"No parser for {spec.name}. Do not store it.")


def stored_error(spec: FieldSpec, value: object) -> str | None:
    """None when a value already in the database is acceptable. 'missing' when blank."""
    if _blank(value):
        return "missing" if spec.mandatory else None
    if spec.name == "skills" and isinstance(value, list) and not value:
        return None
    if spec.name in {"full_name", "college", "branch"}:
        return _string_error(spec, value, reject_placeholders=True)
    if spec.name == "email":
        text = str(value).strip()
        if spec.max_len and len(text) > spec.max_len:
            return f"longer than {spec.max_len} characters"
        if not _EMAIL.fullmatch(text):
            return "not a valid email"
        return None
    if spec.name == "cgpa":
        number = _as_float(value)
        if number is None:
            return "not a number"
        if spec.min_num is not None and (number < spec.min_num or number > (spec.max_num or number)):
            return f"outside {spec.min_num:g}-{spec.max_num:g} on a 10-point scale"
        return None
    if spec.name == "graduation_year":
        year = _as_int(value)
        if year is None:
            return "not an integer year"
        if spec.min_num is not None and not (spec.min_num <= year <= (spec.max_num or year)):
            return f"outside {int(spec.min_num)}-{int(spec.max_num)}"
        return None
    if spec.name == "internship_duration_months":
        number = _as_int(value)
        allowed = {int(item) for item in spec.choices}
        if number not in allowed:
            return f"must be one of {', '.join(spec.choices)}"
        return None
    if spec.name == "available_from":
        parsed = _iso_date(str(value).strip())
        if parsed is None:
            return "not YYYY-MM-DD"
        return _date_range_error(spec, parsed)
    if spec.name in {"degree", "preferred_location"}:
        if str(value).strip() not in spec.choices:
            return f"must be one of {', '.join(spec.choices)}"
        return None
    if spec.name in {"linkedin_url", "github_url"}:
        result = _one_url(spec, str(value).strip())
        return None if result.ok else result.error
    if spec.name == "skills":
        if not isinstance(value, list):
            return "skills must be a list"
        if spec.max_items and len(value) > spec.max_items:
            return f"more than {spec.max_items} skills"
        for item in value:
            if not isinstance(item, str):
                return "each skill must be text"
            error = _item_error(spec, item)
            if error:
                return error
        return None
    if spec.choices and str(value).strip() not in spec.choices:
        return f"must be one of {', '.join(spec.choices)}"
    return None


def values_equal(spec: FieldSpec, left: object, right: object) -> bool:
    if spec.name == "cgpa":
        a, b = _as_float(left), _as_float(right)
        return a is not None and b is not None and abs(a - b) < 0.001
    if spec.name in {"graduation_year", "internship_duration_months", "attempt_count"}:
        return _as_int(left) is not None and _as_int(left) == _as_int(right)
    if spec.name == "skills":
        return _skill_key(left) == _skill_key(right)
    if spec.name in {"email", "linkedin_url", "github_url"}:
        return str(left or "").strip().rstrip("/").casefold() == str(right or "").strip().rstrip("/").casefold()
    return str(left or "").strip() == str(right or "").strip()


def _match_extracted(spec: FieldSpec, proposed: object, extracted: ParseResult, evidence: str) -> ParseResult:
    del evidence
    if not extracted.ok:
        return extracted
    if not _proposed_matches(spec, proposed, extracted.value):
        return ParseResult(
            False,
            error=f"The quote supports {extracted.value!r}, not {proposed!r}.",
        )
    range_error = _range_error(spec, extracted.value)
    if range_error:
        return ParseResult(False, error=range_error)
    return extracted


def _proposed_matches(spec: FieldSpec, proposed: object, canonical: object) -> bool:
    if spec.name in {"linkedin_url", "github_url"}:
        return _canonical_url(spec.name, str(proposed).strip()) == canonical
    if spec.name == "cgpa":
        number = _as_float(proposed)
        return number is not None and abs(number - float(canonical)) < 0.001
    if spec.name in {"graduation_year", "internship_duration_months"}:
        return _as_int(proposed) == _as_int(canonical)
    return str(proposed).strip().casefold() == str(canonical).strip().casefold()


def _range_error(spec: FieldSpec, value: object) -> str | None:
    if spec.name == "cgpa" and spec.min_num is not None:
        number = float(value)
        if number < spec.min_num or number > (spec.max_num or number):
            return f"CGPA must be between {spec.min_num:g} and {spec.max_num:g} on a 10-point scale."
    if spec.name == "graduation_year" and spec.min_num is not None:
        year = int(value)
        if not (spec.min_num <= year <= (spec.max_num or year)):
            return f"Graduation year must be {int(spec.min_num)}-{int(spec.max_num)}."
    if spec.name == "available_from":
        return _date_range_error(spec, str(value))
    if spec.name == "email" and spec.max_len and len(str(value)) > spec.max_len:
        return f"Email is longer than {spec.max_len} characters."
    return None


def _free_text(spec: FieldSpec, proposed: object, evidence: str) -> ParseResult:
    if not isinstance(proposed, str):
        return ParseResult(False, error=f"{spec.name} must be text the candidate said.")
    span = find_span(proposed, evidence)
    if not span:
        return ParseResult(False, error="That text is not in the quote.")
    error = _string_error(spec, span, reject_placeholders=True)
    if error:
        return ParseResult(False, error=error)
    return ParseResult(True, span)


def _string_error(spec: FieldSpec, value: object, reject_placeholders: bool) -> str | None:
    if not isinstance(value, str):
        return "must be text"
    text = re.sub(r"\s+", " ", value).strip()
    if reject_placeholders and text.casefold() in PLACEHOLDERS:
        return "looks like a placeholder, not a real answer"
    min_len = spec.min_len or 1
    max_len = spec.max_len or 10_000
    if len(text) < min_len or len(text) > max_len:
        return f"must be {min_len}-{max_len} characters"
    return None


def _skills(spec: FieldSpec, proposed: object, evidence: str) -> ParseResult:
    items = _skill_items(proposed)
    if isinstance(items, ParseResult):
        return items
    found: list[str] = []
    for item in items:
        span = find_span(item, evidence)
        if not span:
            return ParseResult(False, error=f"{item!r} is not in the quote.")
        error = _item_error(spec, span)
        if error:
            return ParseResult(False, error=error)
        found.append(span)
    if spec.max_items and len(found) > spec.max_items:
        return ParseResult(False, error=f"More than {spec.max_items} skills.")
    return ParseResult(True, found)


def _skill_items(proposed: object) -> list[str] | ParseResult:
    if isinstance(proposed, list):
        if not all(isinstance(item, str) for item in proposed):
            return ParseResult(False, error="Each skill must be text.")
        return [item.strip() for item in proposed if item.strip()]
    if isinstance(proposed, str):
        parts = re.split(r",| and ", proposed)
        return [part.strip() for part in parts if part.strip()]
    return ParseResult(False, error="Skills must be a list of strings.")


def _item_error(spec: FieldSpec, item: str) -> str | None:
    text = item.strip()
    if text.casefold() in PLACEHOLDERS:
        return f"{text!r} is a placeholder skill"
    min_len = spec.item_min or 1
    max_len = spec.item_max or 50
    if len(text) < min_len or len(text) > max_len:
        return f"each skill must be {min_len}-{max_len} characters"
    return None


def _one_email(evidence: str) -> ParseResult:
    found = []
    for match in _EMAIL.findall(evidence):
        lowered = match.casefold()
        if lowered not in found:
            found.append(lowered)
    if not found:
        return ParseResult(False, error="No email address in that quote.")
    if len(found) > 1:
        return ParseResult(False, error=f"Quote contains more than one email: {', '.join(found)}.")
    return ParseResult(True, found[0])


def _one_cgpa(evidence: str) -> ParseResult:
    masked = re.sub(r"(?:out of|/)\s*10\b", " ", evidence, flags=re.IGNORECASE)
    numbers = []
    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", masked):
        numbers.append(float(match.group(1)))
    if not numbers:
        return ParseResult(False, error="No CGPA in that quote. Ask them to say it as a number out of 10.")
    over = [n for n in numbers if n > 10]
    usable = [n for n in numbers if 0 <= n <= 10]
    if over and not usable:
        return ParseResult(
            False,
            error="That score is above 10. Ask for a 10-point CGPA. Do not divide it yourself.",
        )
    unique = []
    for number in usable:
        rounded = float(f"{number:.2f}")
        if rounded not in unique:
            unique.append(rounded)
    if len(unique) != 1:
        return ParseResult(False, error=f"Quote could be more than one CGPA: {unique}.")
    return ParseResult(True, unique[0])


def _one_year(spec: FieldSpec, evidence: str) -> ParseResult:
    if re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", evidence):
        return ParseResult(False, error="That quote is a full date. Ask which year they graduate.")
    low = int(spec.min_num or 2026)
    high = int(spec.max_num or 2028)
    years = []
    for match in re.finditer(r"(?<!\d)(20\d{2})(?!\d)", evidence):
        year = int(match.group(1))
        if year not in years:
            years.append(year)
    if not years:
        return ParseResult(False, error=f"No graduation year in that quote. It has to be {low}-{high}.")
    allowed = [year for year in years if low <= year <= high]
    rejected = [year for year in years if year not in allowed]
    if rejected and not allowed:
        return ParseResult(False, error=f"{rejected[0]} is outside {low}-{high}.")
    if len(allowed) != 1:
        return ParseResult(False, error=f"Quote mentions more than one possible year: {allowed}.")
    return ParseResult(True, allowed[0])


def _one_date(spec: FieldSpec, evidence: str) -> ParseResult:
    found: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in found:
            found.append(value)

    for match in re.finditer(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", evidence):
        add(_iso_date(match.group(0)))
    for match in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})\b", evidence, flags=re.IGNORECASE
    ):
        add(_ymd(int(match.group(3)), _MONTHS.get(match.group(2).casefold()), int(match.group(1))))
    for match in re.finditer(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\,?\s+(\d{4})\b", evidence, flags=re.IGNORECASE
    ):
        add(_ymd(int(match.group(3)), _MONTHS.get(match.group(1).casefold()), int(match.group(2))))
    numeric = list(re.finditer(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", evidence))
    if numeric and not found:
        return ParseResult(
            False,
            error="That date is ambiguous. Ask them to say it as YYYY-MM-DD or like 1 June 2026.",
        )
    if not found:
        return ParseResult(False, error="No start date in that quote.")
    if len(found) > 1:
        return ParseResult(False, error=f"Quote contains more than one date: {', '.join(found)}.")
    error = _date_range_error(spec, found[0])
    if error:
        return ParseResult(False, error=error)
    return ParseResult(True, found[0])


def _one_degree(spec: FieldSpec, evidence: str) -> ParseResult:
    allowed = set(spec.choices)
    folded = re.sub(r"\s+", " ", evidence).strip().casefold()
    bare = _BARE_DEGREE.get(folded.rstrip("."))
    if bare is None and folded in {"b.e", "b.e."}:
        bare = "B.E."
    if bare in allowed and folded in {"be", "b.e", "b.e."}:
        return ParseResult(True, bare)
    hits = []
    for alias, canonical in _DEGREE_ALIASES:
        if canonical not in allowed:
            continue
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", folded):
            if canonical not in hits:
                hits.append(canonical)
    if not hits:
        return ParseResult(False, error=f"Degree must be one of: {', '.join(spec.choices)}.")
    if len(hits) > 1:
        return ParseResult(False, error=f"Quote matches more than one degree: {', '.join(hits)}.")
    return ParseResult(True, hits[0])


def _one_location(spec: FieldSpec, evidence: str) -> ParseResult:
    allowed = set(spec.choices)
    folded = re.sub(r"\s+", " ", evidence).strip().casefold()
    hits = []
    for alias, canonical in _LOCATION_ALIASES:
        if canonical not in allowed:
            continue
        match = re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", folded)
        if not match:
            continue
        prefix = folded[: match.start()]
        if canonical != "No preference" and re.search(r"\b(not|no)\s+$", prefix):
            continue
        if canonical not in hits:
            hits.append(canonical)
    if not hits:
        return ParseResult(False, error=f"Location must be one of: {', '.join(spec.choices)}.")
    if len(hits) > 1:
        return ParseResult(False, error=f"Quote matches more than one location: {', '.join(hits)}.")
    return ParseResult(True, hits[0])


def _one_duration(spec: FieldSpec, evidence: str) -> ParseResult:
    allowed = {int(item) for item in spec.choices}
    found = []
    folded = evidence.casefold()
    for word, number in _DURATION_WORDS.items():
        if number in allowed and re.search(rf"\b{word}\b", folded) and number not in found:
            found.append(number)
    for match in re.finditer(r"(?<![\d.])(\d+)(?![\d.])", evidence):
        number = int(match.group(1))
        if number in allowed and number not in found:
            found.append(number)
    if not found:
        return ParseResult(False, error=f"Duration must be one of: {', '.join(spec.choices)} months.")
    if len(found) > 1:
        return ParseResult(False, error=f"Quote mentions more than one duration: {found}.")
    return ParseResult(True, found[0])


def _one_url(spec: FieldSpec, evidence: str) -> ParseResult:
    raws = re.findall(
        r"https?://[^\s<>]+|(?:www\.)?(?:linkedin|github)\.com/[^\s<>]+",
        evidence,
        flags=re.IGNORECASE,
    )
    parsed = []
    for raw in raws:
        canonical = _canonical_url(spec.name, raw.strip().rstrip(".,);]"))
        if canonical and canonical not in parsed:
            parsed.append(canonical)
    if not parsed:
        kind = "linkedin.com" if spec.name == "linkedin_url" else "github.com profile"
        return ParseResult(False, error=f"No {kind} URL in that quote.")
    if len(parsed) > 1:
        return ParseResult(False, error=f"Quote contains more than one URL: {', '.join(parsed)}.")
    return ParseResult(True, parsed[0])


def _canonical_url(field: str, raw: str) -> str | None:
    text = raw if raw.lower().startswith("http") else "https://" + raw
    try:
        parts = urlparse(text)
    except ValueError:
        return None
    host = (parts.hostname or "").casefold().removeprefix("www.")
    segments = [segment for segment in parts.path.split("/") if segment]
    if field == "linkedin_url":
        if host != "linkedin.com" and not host.endswith(".linkedin.com"):
            return None
        if not segments:
            return None
        path = "/" + "/".join(segments)
        return f"https://{host}{path}"
    if host != "github.com" or len(segments) != 1:
        return None
    user = segments[0]
    if user.casefold() in _GITHUB_RESERVED:
        return None
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", user):
        return None
    return f"https://github.com/{user}"


def _iso_date(text: str) -> str | None:
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None
    if parsed.isoformat() != text:
        return None
    return parsed.isoformat()


def _ymd(year: int, month: int | None, day: int) -> str | None:
    if month is None:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _date_range_error(spec: FieldSpec, iso: str) -> str | None:
    if spec.min_date and iso < spec.min_date:
        return f"Date is before {spec.min_date}."
    if spec.max_date and iso > spec.max_date:
        return f"Date is after {spec.max_date}."
    return None


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _skill_key(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return tuple()
    return tuple(str(item).strip().casefold() for item in value)
