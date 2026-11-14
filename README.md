# ACME Intake Agent

An autonomous chat agent that takes **one candidate ID**, audits the record in the candidate DB, chats to collect what is missing or wrong, validates, writes back safely and records the outcome. Stdlib-only Python (no pip install). See [DECISIONS.md](DECISIONS.md) for the why.

## Setup and run (≤5 commands)
```bash
cp .env.example .env            # then fill CANDIDATE_API_KEY and LLM_API_KEY (Groq free key)
python agent.py --candidate-id c001      # terminal chat; you type as the candidate
python -m acme_agent.local               # web UI at http://127.0.0.1:8080
python -m unittest discover -s tests     # offline tests incl. hallucination guards
python -m harness.run_eval               # live eval (real LLM, in-memory DB)
```
Ending a chat: type `/quit` (candidate leaves), stay silent for `--idle-timeout` seconds (default 300) or press Ctrl-C. The web UI has a "Leave chat" button; closing the page or 5 minutes idle also ends it. Other flags: `--list`, `--reset`. Transcript + tool calls + every API call are saved to `runs/<id>_<time>.json|txt`.

## Deploy on AWS
```bash
aws configure                       # once, in your own terminal (region ap-south-1)
python scripts/aws_tool.py doctor   # checks credentials and .env
python scripts/aws_tool.py deploy   # prints the public HTTPS URL
```
`destroy` removes the stack. Check `<URL>/api/health` -> shows LLM + candidate API status.

## Architecture
```
agent.py / web UI
  -> Conversation.start: GET /v1/candidates/{id} -> audit (missing / invalid / suspicious / auto-fix)
        nothing needed -> no chat (status->completed if the record is already clean)
  -> open(): agent greets and confirms identity by NAME
  -> turn(): LLM  <-->  tools:  confirm_identity | save_fields | end_conversation
        guardrails: grounded evidence, validators, identity gate, overwrite protection,
                    reply scanner, completed-verification (fresh GET)
        writes: PATCH with If-Match + Idempotency-Key, 409 re-read/merge, retry/backoff
  -> end: PATCH status, attempt_count, last_contacted_at, notes (appended)
State: Conversation.state (JSON) = transcript + LLM history + trace; CLI keeps it in memory, web
sends it as an HMAC-signed token. Ground truth is always the DB (re-read before completing).
```
Code: `acme_agent/validators.py` (rules + audit) · `agent.py` (loop, tools, guardrails) · `db.py` · `llm.py` · `web.py` · `cli.py` · `frontend/`.

## Behaviour summary
- **Stops replying / leaves**: `no_answer` (nothing given) or `in_progress` (something saved), attempts +1, note says what is left.
- **Continue later**: asks once for a time; `in_progress` with the time in `notes`.
- **Refuses**: one gentle explanation, then `refused` with the fields listed in `notes`; other data is still saved.
- **Invalid / ambiguous**: validator error is explained; percentage or 4-point CGPA is never converted. Impossible cases -> `needs_human`.
- **Bad existing data**: raised with the candidate; clear junk (`N/A`) treated as missing; format-only fixes auto-applied and noted.
- **Upset / disputed / wrong person**: `needs_human`.

## Status
Built and unit-tested offline (15 tests pass: validators, audit, guardrails, conflicts, outages, abandonment). The candidate API's public endpoints return 200; authenticated calls need your key. Live runs, `runs/` samples and the AWS deploy need `.env` keys and `aws configure`; see the final notes for what has/hasn't been executed. Limitations and next steps: [DECISIONS.md](DECISIONS.md).
