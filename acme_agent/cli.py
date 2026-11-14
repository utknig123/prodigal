"""Terminal chat:  python agent.py --candidate-id c001"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

from acme_agent.agent import Conversation
from acme_agent.config import ROOT, make_clients
from acme_agent.db import DBError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ACME candidate intake agent")
    ap.add_argument("--candidate-id")
    ap.add_argument("--idle-timeout", type=int, default=300, help="seconds of silence before the chat counts as abandoned")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--list", action="store_true", help="list candidates and exit")
    ap.add_argument("--reset", action="store_true", help="reset the workspace and exit")
    ap.add_argument("--check", action="store_true", help="verify CANDIDATE_API_KEY and LLM_API_KEY, then exit")
    a = ap.parse_args(argv)
    db, llm = make_clients()

    if a.check:
        return _check(db, llm)
    try:
        if a.list:
            for c in db.list():
                print(c["id"], c.get("full_name"), c.get("collection_status"))
            return 0
        if a.reset:
            db.reset()
            print("workspace reset")
            return 0
        if not a.candidate_id:
            ap.error("--candidate-id is required")
        if not llm.configured:
            print("LLM_API_KEY is not set (see .env.example)", file=sys.stderr)
            return 2
        conv, audit = Conversation.start(a.candidate_id, db, llm)
    except DBError as e:
        print(f"Database error: {e}", file=sys.stderr)
        return 2

    print(
        f"[agent] {a.candidate_id}: missing={audit.missing} "
        f"invalid={[i['field'] for i in audit.invalid]} "
        f"confirm={[i['field'] for i in audit.confirm]}"
    )
    if conv.s["ended"]:
        print("[agent] Nothing to collect - no conversation started." + (
            " Status set to completed." if conv.s.get("outcome") == "no_action" else ""
        ))
        save_run(conv, a.runs_dir)
        return 0

    print("[agent] Chat started. Type as the candidate. /quit to leave.\n")
    print(f"\nAgent: {conv.open()}\n")
    try:
        while not conv.s["ended"]:
            line = _read("You: ", a.idle_timeout)
            if line is None:
                print("\n[agent] no reply - closing as abandoned")
                conv.abandon("no reply / idle timeout")
                break
            if line.strip().lower() in ("/quit", "/exit"):
                conv.abandon("candidate left the chat (/quit)")
                break
            if not line.strip():
                continue
            print(f"\nAgent: {conv.turn(line)}\n")
    except KeyboardInterrupt:
        conv.abandon("interrupted")
    except Exception as e:
        print(f"[agent] fatal: {e}", file=sys.stderr)
        conv.abandon(f"agent error: {type(e).__name__}")

    if not conv.s.get("final_write_ok", True):
        print("[agent] WARNING: final DB write failed:", conv.s.get("final_write_error"), file=sys.stderr)
    p = save_run(conv, a.runs_dir)
    print(f"\n[agent] outcome={conv.s['outcome']}  run saved to {p}.json")
    return 0


def _check(db, llm) -> int:
    failures = 0
    print(f"Candidate API: {db.base}")
    try:
        schema = db.schema()
        fields = schema.get("fields") if isinstance(schema, dict) else None
        print(f"  schema: ok ({len(fields) if isinstance(fields, list) else '?'} fields)")
    except DBError as e:
        failures += 1
        print(f"  schema: FAILED {e}")
    if not db.key:
        failures += 1
        print("  CANDIDATE_API_KEY: missing")
    else:
        print(f"  CANDIDATE_API_KEY: {_mask(db.key)}")
        try:
            rows = db.list()
            print(f"  key: accepted, {len(rows)} candidates")
        except DBError as e:
            failures += 1
            print(f"  key: REJECTED {e}")

    print(f"LLM: {llm.base_url}  provider={llm.provider}  model={llm.model}")
    if not llm.configured:
        failures += 1
        print("  LLM_API_KEY: missing")
        print(f"{failures} check(s) failed.")
        return 1
    print(f"  LLM_API_KEY: {_mask(llm.api_key)}")
    try:
        models = llm.list_models()
        if models and llm.model not in models:
            print(f"  model list: {llm.model} not listed among {len(models)} models")
            failures += 1
        elif models:
            print("  model list: configured model is listed")
        reply = llm.ping()
        print(f"  completion: ok ({reply[:40]!r})")
        call = llm.probe_tools()
        if call and call.get("name") == "echo_word":
            print(f"  tool calling: ok ({call.get('arguments')})")
        else:
            failures += 1
            print("  tool calling: FAILED")
    except Exception as e:
        failures += 1
        print(f"  LLM check FAILED: {e}")
    print("All checks passed." if not failures else f"{failures} check(s) failed.")
    return 0 if not failures else 1


def _read(prompt: str, timeout: int):
    print(prompt, end="", flush=True)
    lines: queue.Queue = queue.Queue()

    def pump():
        try:
            lines.put(sys.stdin.readline())
        except Exception:
            lines.put("")

    threading.Thread(target=pump, daemon=True).start()
    try:
        line = lines.get(timeout=timeout if timeout and timeout > 0 else None)
    except queue.Empty:
        return None
    if line == "":
        return None
    return line.rstrip("\r\n")


def save_run(conv, runs_dir) -> Path:
    d = Path(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = d / f"{conv.s['cid']}_{stamp}"
    state = conv.public_state()
    base.with_suffix(".json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    lines = []
    for item in state.get("transcript") or []:
        role = item.get("role", "?")
        lines.append(f"{role}: {item.get('text', '')}")
    lines.append("")
    lines.append(f"outcome={state.get('outcome')}")
    lines.append(f"saved={json.dumps(state.get('saved'), default=str)}")
    base.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    return base


def _mask(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-4:]} ({len(secret)} chars)"


if __name__ == "__main__":
    raise SystemExit(main())
