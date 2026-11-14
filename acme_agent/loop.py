"""One candidate, one conversation. The model speaks; this module decides the outcome."""

from __future__ import annotations

import json
import re
from datetime import datetime

from acme_agent.assess import assess
from acme_agent.client import ApiError
from acme_agent.llm import LLMError, LLMTurn, assistant_message
from acme_agent.prompts import TOOLS, system_prompt
from acme_agent.schema_rules import Schema
from acme_agent.tools import Session, Toolbox

MAX_ROUNDS = 40
TIMEOUT = object()


def run_collection(client, model, candidate_id: str, input_fn, output_fn, trace_fn, transcript) -> dict:
    schema = client.get_schema()
    record = client.get_candidate(candidate_id)
    assessment = assess(record, schema)
    if assessment.unwritable:
        note = _line(f"Read-only problem: {', '.join(assessment.unwritable)}. No chat opened.")
        record = _write_outcome(client, record, "needs_human", note, schema)
        output_fn("This record has a read-only field the agent cannot fix. Marked needs_human.")
        return _result(record, "needs_human", False)

    if assessment.ready:
        if record.get("collection_status") != "completed":
            note = _line("Schema check passed. No chat opened.")
            record = _write_outcome(client, record, "completed", note, schema)
        name = record.get("full_name") or candidate_id
        output_fn(f"{name} already has a complete, valid application. No chat opened.")
        return _result(record, "completed", False)

    record = _mark_contact(client, record, assessment, schema)
    if record.get("collection_status") == "needs_human" and int(record.get("attempt_count") or 0) >= 50:
        output_fn("Attempt limit reached. Marked needs_human without chatting.")
        return _result(record, "needs_human", False)

    session = Session()
    box = Toolbox(client, record, schema, session)
    problems = _problems(schema, record, assessment)
    valid = _valid(schema, record, assessment)
    messages = [
        {"role": "system", "content": system_prompt(str(record.get("full_name") or "there"), valid, problems)},
        {
            "role": "user",
            "content": (
                "Program note, not the candidate: begin. Greet them by the name on file and "
                "ask if that is who you are speaking with. Do not ask for application data yet."
            ),
        },
    ]
    trace_fn("Commands: /quit stops, /later continues another time, /human asks for a person.")
    nudges = 0
    for _ in range(MAX_ROUNDS):
        try:
            turn = model.complete(messages, TOOLS)
        except LLMError as exc:
            outcome = "in_progress"
            note = _summary(outcome, session, assess(box.record, schema), "", f"model_error={str(exc)[:200]}")
            record = _write_outcome(client, box.record, outcome, note, schema)
            output_fn("The model call failed. I saved the status and stopped.")
            trace_fn(f"[model-error] {exc}")
            return _result(record, outcome, True)
        if turn.tool_calls:
            messages.append(assistant_message(turn))
            stop = _apply_tools(box, turn, messages, transcript, trace_fn)
            if stop:
                return _close(client, box, schema, stop, output_fn, transcript)
            continue
        text = (turn.text or "").strip()
        if not text or _looks_like_json(text):
            nudges += 1
            messages.append(assistant_message(turn))
            if nudges > 2:
                note = _summary("needs_human", session, assess(box.record, schema), "", "reason=model_sent_no_message")
                record = _write_outcome(client, box.record, "needs_human", note, schema)
                output_fn("Stopping because the model did not produce a message.")
                return _result(record, "needs_human", True)
            messages.append(
                {
                    "role": "user",
                    "content": "Program note, not the candidate: that was not shown. Say one short message or call a tool.",
                }
            )
            continue
        nudges = 0
        output_fn(text)
        _event(transcript, {"type": "assistant", "text": text})
        messages.append({"role": "assistant", "content": text})
        session.last_assistant = text
        reply = _read(input_fn)
        if reply is None:
            return _close_command(client, box, schema, "eof", output_fn, transcript)
        if reply is TIMEOUT:
            _event(transcript, {"type": "command", "text": "idle_timeout"})
            return _close_command(client, box, schema, "idle_timeout", output_fn, transcript)
        if not reply.strip():
            output_fn("I didn't catch that. Could you say it again?")
            continue
        command = reply.strip().casefold()
        if command in {"/quit", "/later", "/human"}:
            _event(transcript, {"type": "command", "text": command})
            return _close_command(client, box, schema, command, output_fn, transcript)
        session.candidate_messages.append(reply.strip())
        messages.append({"role": "user", "content": reply.strip()})
        _event(transcript, {"type": "candidate", "text": reply.strip()})

    outcome = "in_progress" if session.saved or session.candidate_messages else "no_answer"
    note = _summary(outcome, session, assess(box.record, schema), "", "reason=turn_limit")
    record = _write_outcome(client, box.record, outcome, note, schema)
    output_fn(_closing(outcome, str(record.get("full_name") or "")))
    return _result(record, outcome, True)


def _apply_tools(box: Toolbox, turn: LLMTurn, messages: list[dict], transcript, trace_fn) -> dict | None:
    stop = None
    for call in turn.tool_calls:
        if call.arguments.get("_invalid_json"):
            result = {"ok": False, "error": "Tool arguments were not valid JSON."}
        else:
            try:
                result = box.dispatch(call.name, call.arguments)
            except ApiError as exc:
                result = {
                    "ok": False,
                    "error": f"The database call failed ({exc.status}). Nothing new was confirmed. "
                    "Tell the candidate briefly and try once more, or finish with continue_later.",
                }
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
        _event(transcript, {"type": "tool", "name": call.name, "arguments": call.arguments, "result": result})
        _trace_result(trace_fn, result)
        if result.get("stop"):
            stop = result
    return stop


def _close(client, box: Toolbox, schema: Schema, stop: dict, output_fn, transcript) -> dict:
    outcome = str(stop["outcome"])
    note = _summary(
        outcome,
        box.session,
        assess(box.record, schema),
        str(stop.get("quote") or ""),
        str(stop.get("reason") or ""),
    )
    record = _write_outcome(client, box.record, outcome, note, schema)
    output_fn(_closing(outcome, str(record.get("full_name") or "")))
    _event(transcript, {"type": "outcome", "outcome": outcome})
    return _result(record, outcome, True)


def _close_command(client, box: Toolbox, schema: Schema, command: str, output_fn, transcript) -> dict:
    session = box.session
    if command == "/human":
        outcome = "needs_human"
    elif command == "/later":
        outcome = "in_progress"
    elif session.identity == "rejected":
        outcome = "needs_human"
    elif session.refused_mandatory:
        outcome = "refused"
    elif session.saved:
        outcome = "in_progress"
    else:
        outcome = "no_answer"
    reason = f"command={command}"
    note = _summary(outcome, session, assess(box.record, schema), "", reason)
    record = _write_outcome(client, box.record, outcome, note, schema)
    output_fn(_closing(outcome, str(record.get("full_name") or "")))
    _event(transcript, {"type": "outcome", "outcome": outcome})
    return _result(record, outcome, True)


def _mark_contact(client, record: dict, assessment, schema: Schema) -> dict:
    spec = schema.spec("attempt_count")
    limit = int(spec.max_num if spec.max_num is not None else 50)
    count = int(record.get("attempt_count") or 0)
    if count >= limit:
        note = _line(f"attempt_count is {count}, which is the limit. No chat opened.")
        return _write_outcome(client, record, "needs_human", note, schema)
    blocking = ", ".join(assessment.blocking) or "none"
    note = _append(record.get("notes"), _line(f"Opened a chat. Blocking: {blocking}."), _note_limit(schema))
    fields = {
        "attempt_count": count + 1,
        "last_contacted_at": _stamp(),
        "collection_status": "in_progress",
        "notes": note,
    }
    return client.patch_retry(record, fields)


def _write_outcome(client, record: dict, outcome: str, note: str, schema: Schema) -> dict:
    if outcome not in schema.statuses:
        raise ApiError(0, f"Refusing to write unknown status {outcome}")
    fields = {
        "collection_status": outcome,
        "notes": _append(record.get("notes"), note, _note_limit(schema)),
    }
    return client.patch_retry(record, fields)


def _problems(schema: Schema, record: dict, assessment) -> list[dict]:
    rows = []
    for name in assessment.missing:
        rows.append({"field": name, "problem": "missing", "rules": schema.spec(name).rules})
    for name, reason in list(assessment.invalid) + list(assessment.optional_invalid):
        rows.append(
            {"field": name, "problem": reason, "on_file": record.get(name), "rules": schema.spec(name).rules}
        )
    return rows


def _valid(schema: Schema, record: dict, assessment) -> dict:
    blocked = set(assessment.blocking)
    valid = {}
    for name in schema.profile_fields:
        if name in blocked:
            continue
        value = record.get(name)
        if value not in (None, "", []):
            valid[name] = value
    return valid


def _summary(outcome: str, session: Session, assessment, quote: str, reason: str) -> str:
    parts = [f"outcome={outcome}"]
    parts.append("saved=" + (",".join(session.saved) if session.saved else "none"))
    parts.append(
        "refused=" + (",".join(session.refused_mandatory) if session.refused_mandatory else "none")
    )
    parts.append("remaining=" + (",".join(assessment.blocking) if assessment.blocking else "none"))
    if reason:
        parts.append(f"reason={reason}")
    if quote:
        parts.append('quote="' + quote.replace('"', "'")[:300] + '"')
    return _line("; ".join(parts))


def _closing(outcome: str, name: str) -> str:
    who = name or "there"
    lines = {
        "completed": f"Thanks {who}. That's everything we needed. Your application is updated.",
        "in_progress": f"Thanks {who}. We'll continue another time. What's still missing stays on the application.",
        "no_answer": f"Ending the chat. Nothing new was saved for {who}.",
        "refused": f"Understood, {who}. I won't keep asking for what you don't want to share.",
        "needs_human": f"Thanks {who}. I'm handing this to a person on the team.",
    }
    return lines[outcome]


def _trace_result(trace_fn, result: dict) -> None:
    for item in result.get("saved") or []:
        if item.get("wrote"):
            trace_fn(f"[saved] {item['field']} = {item['stored']}")
        else:
            trace_fn(f"[kept] {item['field']}")
    for item in result.get("errors") or []:
        trace_fn(f"[rejected] {item.get('field')}: {item.get('error')}")
    if result.get("identity"):
        trace_fn(f"[identity] {result['identity']}")
    if result.get("refused"):
        trace_fn(f"[refused] {result['refused']}")
    if result.get("error") and not result.get("errors"):
        trace_fn(f"[tool] {result['error']}")
    if result.get("stop"):
        trace_fn(f"[finish] {result.get('outcome')}")


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("```"):
        return True
    if stripped.startswith("{") and any(token in stripped for token in ('"field"', '"intent"', '"updates"')):
        return True
    return False


def _read(input_fn):
    try:
        return input_fn("You: ")
    except EOFError:
        return None
    except TimeoutError:
        return TIMEOUT
    except KeyboardInterrupt:
        return "/quit"


def _append(existing: object, entry: str, limit: int) -> str:
    old = existing if isinstance(existing, str) else ""
    entry = re.sub(r"\s+", " ", entry).strip()
    combined = f"{old.rstrip()}\n{entry}".strip() if old.strip() else entry
    if len(combined) <= limit:
        return combined
    if len(entry) >= limit:
        return entry[-limit:]
    return combined[-limit:]


def _note_limit(schema: Schema) -> int:
    return schema.spec("notes").max_len or 4000


def _stamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _line(text: str) -> str:
    return f"[{_stamp()}] {text}"


def _event(transcript, event: dict) -> None:
    if transcript is not None:
        transcript.add(event)


def _result(record: dict, outcome: str, chatted: bool) -> dict:
    return {"candidate_id": record.get("id"), "outcome": outcome, "record": record, "chatted": chatted}
