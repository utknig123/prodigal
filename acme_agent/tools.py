"""Tools the model may call. Every write is checked here before it reaches the API."""

from __future__ import annotations

import re

from acme_agent.assess import assess
from acme_agent.parse import interpret, stored_error, values_equal
from acme_agent.quotes import contains_quote
from acme_agent.schema_rules import Schema

AFFIRM = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|speaking|haan|i am|i'm|this is me|that's me|thats me)\b",
    re.IGNORECASE,
)
DENY = re.compile(
    r"\b(no|nope|nah|wrong person|not me|someone else)\b",
    re.IGNORECASE,
)
REFUSE = re.compile(
    r"\b(won't|wont|will not|don't want|do not want|dont want|rather not|prefer not|"
    r"refuse|not comfortable|can't share|cannot share|private|skip)\b",
    re.IGNORECASE,
)
SHORT_NO = frozenset({"no", "nope", "nah", "no thanks", "no thank you"})
HUMAN_REASONS = frozenset({"contradiction", "upset", "outside_rules", "other"})
FIELD_WORDS = {
    "full_name": ("name",),
    "email": ("email", "e-mail"),
    "college": ("college", "university", "institute"),
    "degree": ("degree",),
    "branch": ("branch", "specialisation", "specialization", "stream", "major"),
    "graduation_year": ("graduation", "graduate", "graduating", "year"),
    "cgpa": ("cgpa", "gpa", "grade"),
    "preferred_location": ("location", "city", "mumbai", "bengaluru", "bangalore", "remote"),
    "available_from": ("available", "start date", "join", "joining", "date"),
    "internship_duration_months": ("duration", "months", "how long"),
    "linkedin_url": ("linkedin",),
    "github_url": ("github",),
    "skills": ("skill",),
}


class Session:
    def __init__(self) -> None:
        self.candidate_messages: list[str] = []
        self.last_assistant = ""
        self.identity = "unknown"
        self.saved: list[str] = []
        self.refused_mandatory: list[str] = []
        self.skipped_optional: list[str] = []


class Toolbox:
    def __init__(self, client, record: dict, schema: Schema, session: Session):
        self.client = client
        self.record = record
        self.schema = schema
        self.session = session

    def dispatch(self, name: str, arguments: dict) -> dict:
        handler = {
            "get_progress": self.get_progress,
            "confirm_identity": self.confirm_identity,
            "confirm_on_file": self.confirm_on_file,
            "record_fields": self.record_fields,
            "refuse_field": self.refuse_field,
            "finish": self.finish,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool {name}."}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "Tool arguments must be an object."}
        return handler(arguments)

    def get_progress(self, _arguments: dict) -> dict:
        self.record = self.client.get_candidate(str(self.record["id"]))
        return {"ok": True, **self._snapshot()}

    def confirm_identity(self, arguments: dict) -> dict:
        matched = _as_bool(arguments.get("matched"))
        evidence = str(arguments.get("evidence") or "")
        if matched is None:
            return {"ok": False, "error": "matched must be true or false."}
        quote_error = self._quote(evidence)
        if quote_error:
            return {"ok": False, "error": quote_error}
        name = str(self.record.get("full_name") or "").strip()
        if name and name.casefold() not in self.session.last_assistant.casefold():
            return {
                "ok": False,
                "error": "Your last message did not use the name on file. Ask if you are speaking with that person.",
            }
        said_yes = bool(AFFIRM.search(evidence))
        said_no = bool(DENY.search(evidence))
        if said_yes and said_no:
            return {"ok": False, "error": "That quote sounds like both a yes and a no. Ask again."}
        if matched and not said_yes:
            return {"ok": False, "error": "matched true needs a clear yes in the candidate's quote."}
        if not matched and not said_no:
            return {"ok": False, "error": "matched false needs a clear no in the candidate's quote."}
        if matched:
            self.session.identity = "confirmed"
            return {"ok": True, "identity": "confirmed", **self._snapshot()}
        self.session.identity = "rejected"
        return {
            "ok": True,
            "identity": "rejected",
            "stop": True,
            "outcome": "needs_human",
            "reason": "wrong_person",
            "quote": evidence,
        }

    def confirm_on_file(self, arguments: dict) -> dict:
        field_name = str(arguments.get("field") or "")
        evidence = str(arguments.get("evidence") or "")
        if field_name not in self.schema.profile_fields:
            return {"ok": False, "error": f"{field_name} is not a profile field you can confirm."}
        blocked = self._identity_block(field_name)
        if blocked:
            return {"ok": False, "error": blocked}
        quote_error = self._quote(evidence)
        if quote_error:
            return {"ok": False, "error": quote_error}
        if not AFFIRM.search(evidence):
            return {"ok": False, "error": "The quote does not clearly confirm the value."}
        spec = self.schema.spec(field_name)
        current = self.record.get(field_name)
        error = stored_error(spec, current)
        if error:
            return {"ok": False, "error": f"The value on file is not acceptable ({error}). Ask for a replacement."}
        if not _value_shown(spec, current, self.session.last_assistant):
            return {
                "ok": False,
                "error": "Your last message did not state the value on file, so a yes is not proof.",
            }
        return {"ok": True, "kept": field_name, "wrote": False, **self._snapshot()}

    def record_fields(self, arguments: dict) -> dict:
        updates = arguments.get("updates")
        if not isinstance(updates, list) or not updates:
            return {"ok": False, "error": "updates must be a non-empty list."}
        accepted = []
        errors = []
        to_write: dict = {}
        for item in updates:
            if not isinstance(item, dict):
                errors.append({"field": None, "error": "Each update must be an object."})
                continue
            field_name = str(item.get("field") or "")
            result = self._prepare(field_name, item.get("value"), str(item.get("evidence") or ""))
            if not result["ok"]:
                errors.append({"field": field_name, "error": result["error"]})
                continue
            if result.get("unchanged"):
                accepted.append({"field": field_name, "stored": result["stored"], "wrote": False})
                continue
            to_write[field_name] = result["stored"]
            accepted.append({"field": field_name, "stored": result["stored"], "wrote": True})
        if to_write:
            self.record = self.client.patch_retry(self.record, to_write)
            for field_name in to_write:
                if field_name not in self.session.saved:
                    self.session.saved.append(field_name)
                if field_name in self.session.refused_mandatory:
                    self.session.refused_mandatory.remove(field_name)
        return {"ok": not errors, "saved": accepted, "errors": errors, **self._snapshot()}

    def refuse_field(self, arguments: dict) -> dict:
        field_name = str(arguments.get("field") or "")
        evidence = str(arguments.get("evidence") or "")
        if field_name not in self.schema.profile_fields:
            return {"ok": False, "error": f"{field_name} is not a profile field."}
        quote_error = self._quote(evidence)
        if quote_error:
            return {"ok": False, "error": quote_error}
        if not _is_refusal(evidence):
            return {"ok": False, "error": "That quote is not a refusal. Ask again or keep collecting."}
        mentioned = fields_mentioned(self.session.last_assistant)
        if field_name not in mentioned:
            return {"ok": False, "error": "The last question did not ask for this field."}
        if len(mentioned) > 1:
            return {"ok": False, "error": "The last question asked for more than one field. Ask about one, then record the refusal."}
        spec = self.schema.spec(field_name)
        if spec.mandatory:
            if field_name not in self.session.refused_mandatory:
                self.session.refused_mandatory.append(field_name)
        else:
            if field_name not in self.session.skipped_optional:
                self.session.skipped_optional.append(field_name)
            if stored_error(spec, self.record.get(field_name)):
                self.record = self.client.patch_retry(self.record, {field_name: None})
                if not _blank(self.record.get(field_name)):
                    return {
                        "ok": False,
                        "error": "The database kept an invalid optional value after a clear.",
                        "stop": True,
                        "outcome": "needs_human",
                        "reason": "outside_rules",
                        "quote": evidence,
                    }
        return {"ok": True, "refused": field_name, **self._snapshot()}

    def finish(self, arguments: dict) -> dict:
        intent = str(arguments.get("intent") or "").strip().casefold()
        quote = str(arguments.get("quote") or "")
        reason = str(arguments.get("reason") or "").strip().casefold()
        if quote:
            quote_error = self._quote(quote)
            if quote_error:
                return {"ok": False, "error": quote_error}
        assessment = assess(self.record, self.schema)
        if intent == "complete":
            if self.session.identity != "confirmed":
                return {"ok": False, "error": "Identity is not confirmed."}
            if self.session.refused_mandatory:
                return {"ok": False, "error": "A mandatory field was refused. finish with refused, not complete."}
            if not assessment.ready:
                return {"ok": False, "error": "The record is still incomplete.", **self._snapshot()}
            return {"ok": True, "stop": True, "outcome": "completed", "quote": quote}
        if intent == "continue_later":
            return {"ok": True, "stop": True, "outcome": "in_progress", "quote": quote}
        if intent == "refused":
            if not self.session.refused_mandatory:
                return {"ok": False, "error": "Call refuse_field for a mandatory field before finish refused."}
            return {"ok": True, "stop": True, "outcome": "refused", "quote": quote}
        if intent == "needs_human":
            if reason not in HUMAN_REASONS:
                return {"ok": False, "error": "reason must be contradiction, upset, outside_rules, or other."}
            if not quote:
                return {"ok": False, "error": "needs_human requires a verbatim candidate quote."}
            return {"ok": True, "stop": True, "outcome": "needs_human", "reason": reason, "quote": quote}
        return {"ok": False, "error": "intent must be complete, continue_later, refused, or needs_human."}

    def _prepare(self, field_name: str, proposed: object, evidence: str) -> dict:
        if field_name not in self.schema.profile_fields:
            return {"ok": False, "error": f"{field_name} cannot be written by the agent."}
        blocked = self._identity_block(field_name)
        if blocked:
            return {"ok": False, "error": blocked}
        quote_error = self._quote(evidence)
        if quote_error:
            return {"ok": False, "error": quote_error}
        spec = self.schema.spec(field_name)
        parsed = interpret(spec, proposed, evidence)
        if not parsed.ok:
            return {"ok": False, "error": parsed.error}
        stored = parsed.value
        if field_name == "skills":
            stored = _merge_skills(self.record.get("skills"), stored, spec.max_items or 30)
            if isinstance(stored, str):
                return {"ok": False, "error": stored}
        if values_equal(spec, self.record.get(field_name), stored):
            return {"ok": True, "unchanged": True, "stored": self.record.get(field_name)}
        return {"ok": True, "stored": stored}

    def _identity_block(self, field_name: str) -> str | None:
        if self.session.identity == "confirmed":
            return None
        name_error = stored_error(self.schema.spec("full_name"), self.record.get("full_name"))
        if field_name == "full_name" and name_error:
            return None
        return "Confirm identity before recording this. Call confirm_identity."

    def _quote(self, evidence: str) -> str | None:
        if not contains_quote(evidence, self.session.candidate_messages):
            return "evidence is not a verbatim quote from the candidate."
        return None

    def _snapshot(self) -> dict:
        assessment = assess(self.record, self.schema)
        return {
            "blocking": _blocking(self.schema, self.record, assessment),
            "identity": self.session.identity,
            "hint": _hint(self.session, assessment),
        }


def fields_mentioned(message: str) -> list[str]:
    found = []
    for field_name, words in FIELD_WORDS.items():
        if any(
            re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", message, flags=re.IGNORECASE)
            for word in words
        ):
            found.append(field_name)
    return found


def _is_refusal(evidence: str) -> bool:
    if REFUSE.search(evidence):
        return True
    return evidence.strip().casefold() in SHORT_NO


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        folded = value.strip().casefold()
        if folded in {"true", "yes"}:
            return True
        if folded in {"false", "no"}:
            return False
    return None


def _value_shown(spec, value: object, message: str) -> bool:
    forms = {str(value).strip()}
    if spec.name == "cgpa":
        number = float(value)
        forms.add(f"{number:.2f}")
        forms.add(f"{number:g}")
    if spec.name == "internship_duration_months":
        forms.add(str(int(value)))
    folded = message.casefold()
    return any(form.casefold() in folded for form in forms if form)


def _merge_skills(existing: object, new_items: object, limit: int) -> list[str] | str:
    current = list(existing) if isinstance(existing, list) else []
    merged: list[str] = [str(item) for item in current]
    seen = {item.strip().casefold() for item in merged}
    for item in new_items if isinstance(new_items, list) else []:
        key = item.strip().casefold()
        if key not in seen:
            merged.append(item)
            seen.add(key)
    if len(merged) > limit:
        return f"That would store {len(merged)} skills; the limit is {limit}."
    return merged


def _blocking(schema: Schema, record: dict, assessment) -> list[dict]:
    rows = []
    for name in assessment.missing:
        rows.append({"field": name, "problem": "missing", "rules": schema.spec(name).rules})
    for name, reason in assessment.invalid + assessment.optional_invalid:
        rows.append(
            {
                "field": name,
                "problem": reason,
                "on_file": record.get(name),
                "rules": schema.spec(name).rules,
            }
        )
    return rows


def _hint(session: Session, assessment) -> str:
    if session.identity == "rejected":
        return "Stop. The program will mark needs_human."
    if assessment.ready and session.identity == "confirmed" and not session.refused_mandatory:
        return "Nothing is blocking. Thank them and finish with intent complete."
    if session.refused_mandatory:
        return "A mandatory field was refused. Collect anything else they will give, then finish with refused."
    if session.identity != "confirmed":
        return "Confirm identity before recording application fields."
    return "Ask about one blocking field."


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and not value:
        return True
    return False
