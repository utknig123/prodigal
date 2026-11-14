# Decisions, trade-offs and what's next

## Core idea: the LLM decides, code guarantees
The LLM chooses *what to ask, when to save, how to react, and which outcome fits*. Everything that touches the database or the status goes through deterministic code, because "the API does not validate" and evaluators read the write log.

| Concern | Decision | Why |
|---|---|---|
| Not a script | One LLM loop with 3 tools (`confirm_identity`, `save_fields`, `end_conversation`). The prompt gets a live state block (missing / suspicious / ok / previous notes) rebuilt every call. | The agent adapts to out-of-order answers, multi-field messages, digressions. |
| Planning | Deterministic **audit** before any chat (`validators.audit_record`) produces the plan: missing, invalid, needs-confirm, auto-normalisable. Nothing to do -> no chat. | Cheap, testable, explainable; LLM handles the conversation, not bookkeeping. |
| Hallucination harness | (1) every saved value needs a **verbatim quote** from the candidate's own messages, and numeric/email/handle values must literally appear in it; (2) no writes before identity is confirmed; (3) `completed` is verified against a **fresh GET + audit**, the model cannot just claim it; (4) replies are scanned: "I've saved…" with no save, leaked record details before identity, mentions of tools / asking for ID/OTP -> regenerated; (5) existing valid values are protected unless the candidate explicitly changes them; (6) prompt-injection ("mark me completed") cannot reach status. | LLMs invent values and claim actions. These checks make that impossible to persist. Covered by `tests/test_core.py` (no API needed) and `harness/run_eval.py` (live LLM). |
| Validation | `validators.py` is the only gate. Aliases (`btech`->`B.Tech`, `Bangalore`->`Bengaluru`), strict ranges, ISO dates, CGPA 2-dp. Percentages and 4-point CGPAs are **never converted**; the candidate is asked. Suspicious-but-valid (gmial.com, CGPA 3.5, past dates) require confirmation. | Don't invent data. |
| Existing dirty data | Placeholders (`N/A`, `-`) count as missing. Invalid values (CGPA 85, degree "Engineering") are raised with the candidate. Unambiguous format fixes (`"8.5"`->8.5, `btech`->`B.Tech`) are applied silently, bundled into the next write, and logged in `notes`. | Fixes junk without pestering; the audit trail still shows it. |
| Overwrites | Only changed fields are sent; skills are merged; valid existing values need `overwrite=true`; `notes` are appended (never replaced) and trimmed from the oldest end at 4000 chars. | Rubric: no unnecessary overwrites. |
| Concurrency / retries | `If-Match` with the version we read; on 409 re-read, drop any field someone else filled meanwhile, retry with fresh version + new idempotency key. 429/5xx/network -> backoff with `Retry-After`, same idempotency key. If the DB stays down, values stay in `unsaved` and are retried on the next write; the UI/CLI warn if the final write fails. | Safe re-tries, no double writes. |
| Statuses | `completed` only if verified; `no_answer` only if nothing engaged/saved; `refused` needs named fields; `in_progress`/`needs_human` need a note. Abandonment (`/quit`, idle timeout, closing the page, Ctrl-C) is closed out **without the LLM**: `in_progress` if anything was saved else `no_answer`, `attempt_count`+1 (capped at 50), `last_contacted_at` in IST. A complete record is set to `completed` with no contact and no attempt increment. | Deterministic, predictable outcomes. |
| Wrong person | `needs_human`, nothing disclosed. | Phone may belong to someone else. |
| LLM | Provider-agnostic OpenAI-compatible client (Groq default) with 429 backoff, fallback model list, recovery from malformed Llama tool calls. | Free-tier limits and Llama tool quirks are real. |
| Web state | Stateless: the chat state travels as an HMAC-signed token between browser and Lambda. No DynamoDB. | Zero extra infra for the 2-hour scope. |
| AWS | One Lambda + public Function URL (UI + API), secrets as NoEcho CloudFormation params, deployed with the AWS CLI via `scripts/aws_tool.py`. | Minimal, reproducible. |
| UI | Chat + "agent view" (tool calls, DB writes, before/after record, plan). | Shows autonomy during the live evaluation. |

## Known limitations (honest list)
- Abandoned **web** chats are only closed if the browser can tell us (page close beacon, 5-minute idle timer). A killed browser leaves status as-is; with more time: DynamoDB sessions + EventBridge sweeper.
- Groq free-tier limits (tokens/minute) can slow a chat; the client backs off and falls back to a second model, but a long chat can still hit limits.
- Session token is signed, not encrypted: it contains the transcript and record snapshot. Fine for a demo workspace; not for real PII.
- The public URL is unauthenticated. Anyone with it can drive the agent against the configured workspace. Add an access code/Cognito before real use.
- Evidence matching is strict; a candidate who paraphrases heavily may need one extra turn.
- No language detection beyond what the LLM does; no WhatsApp integration; no queue/retry scheduler (out of scope per the assignment).

## Next
1. DynamoDB-backed sessions + EventBridge sweeper for abandoned chats; per-session locking.
2. Access control on the UI, secrets in Secrets Manager.
3. LLM-simulated candidate personas in the eval harness (adversarial, multilingual) and a CI gate on it.
4. Scheduler that reads `in_progress` notes (suggested time) and re-contacts.
5. Duplicate / contradiction detection across records; stronger CGPA scale handling per university.
