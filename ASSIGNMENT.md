# Assignment NMIMS'26: AI Agent for Candidate Information Collection

**Format:** individual · **Time:** 2 hours · **Evaluation:** live, with an evaluator

## Problem statement

**ACME Corp** launched a Google Form to collect internship applications. Several fields that
should have been mandatory were accidentally marked optional, so many applications are incomplete.

You can rely on one thing: every candidate gave a valid phone number, verified by OTP, so you can
reach them over chat (think WhatsApp). For this assignment the chat runs locally; no messaging
integration is needed.

Build an **autonomous AI chat agent** that takes **one candidate** and drives their application to
completion in **a single chat conversation**:

1. **Work out what's needed.** Given a candidate ID, fetch the record and decide what to collect.
   Look at the existing data too, not only at what's missing. If nothing is needed, don't start a
   conversation.
2. **Start a chat with the candidate.** Greet them and confirm you're talking to the right person.
3. **Hold a natural conversation** to collect the missing information. Handle follow-up
   questions, ambiguous answers and incomplete answers.
4. **Handle exceptions:** the candidate stops replying or leaves, asks to continue later, gives
   invalid information, or won't share something.
5. **Update the database.** Validate what you collected and write it back without needlessly
   overwriting existing information.
6. **Record the outcome** of the conversation: completed, follow-up pending, unsuccessful, or
   needs a human.

We want to see **agentic behaviour**: an LLM that decides, calls tools, and recovers from
problems. A fixed question-by-question script or form is not what we're looking for. You may use
any stack, LLM and agent framework.

### Expected flow

```
run agent with --candidate-id <id>
  → GET /v1/candidates/{id}            the agent now knows who it's talking to and what's missing
  → start the chat                     confirm identity by NAME. Never ask the candidate for an ID
  → converse, collect, validate
  → PATCH /v1/candidates/{id}          data + collection_status. Saving progress mid-chat is fine
  → exit
```

Your agent is **not** expected to work through a queue, retry, or follow up later itself. It
records what happened (e.g. `in_progress` with a note on what's left, or `no_answer`) and stops.

> Two hours is short. A submission that works end to end, plus clear thinking about the edge
> cases, beats a half-built system. Tell us what you'd do next.

---

## 1. The candidate database

**Base URL:** `https://candidate-db-api.service.external.usea2.aws.prodigaltech.com`

You'll get a **personal API key**. Your key gives you a private workspace of 8 candidates. Nobody
else can see or change your data, and you can't see theirs.

Interactive API docs are at `<BASE_URL>/docs`. Click **Authorize** and paste your key.

### Endpoints

All requests need the header `Authorization: Bearer <your_api_key>`.

| Method | Path | What it does |
|---|---|---|
| `GET` | `/v1/schema` | Field definitions and the rules your agent must enforce (no key needed) |
| `GET` | `/v1/candidates` | List your candidates, to find IDs to test with. Filter: `?collection_status=...` |
| `GET` | `/v1/candidates/{id}` | Get one candidate |
| `PATCH` | `/v1/candidates/{id}` | Update fields. **Only the fields you send are changed** |
| `POST` | `/v1/reset` | Restore your workspace to the original data. Use it freely while testing |

Every candidate record also includes `version` and `updated_at`. **Working out which fields are
missing is up to you**, using the schema below.

**Always identify a record by its `id`** (e.g. `c001`), not by name or phone number.

### Schema

> **The API does not validate field values.** It stores whatever you send, including a bad
> email, a CGPA of 85 or an unknown degree. Enforcing the rules below is your agent's job, and
> the evaluators will check what ended up in the database.

**Mandatory fields** (a record is complete when all of them are filled):

| Field | Type | Rules |
|---|---|---|
| `full_name` | string | 2–100 chars. Always present |
| `phone` | string (E.164) | **Read-only.** OTP-verified |
| `email` | string | Valid email address |
| `college` | string | 2–150 chars |
| `degree` | enum | `B.Tech`, `B.E.`, `B.Sc`, `BCA`, `MBA Tech`, `MCA`, `M.Tech`, `Other` |
| `branch` | string | Specialisation, e.g. `Computer Engineering` |
| `graduation_year` | integer | 2026–2028 |
| `cgpa` | number | 0.0–10.0 (10-point scale), stored to 2 decimals |
| `preferred_location` | enum | `Mumbai`, `Bengaluru`, `Remote`, `No preference` |
| `available_from` | date | `YYYY-MM-DD`, between 2026-01-01 and 2027-12-31 |
| `internship_duration_months` | integer enum | `2`, `3`, `6` |

**Optional fields:** `linkedin_url` (a linkedin.com URL), `github_url` (a github.com profile URL),
`skills` (up to 30 strings).

**Collection tracking.** These fields are yours to maintain:

| Field | Type | Rules |
|---|---|---|
| `collection_status` | enum | `pending` (default), `in_progress`, `completed`, `no_answer`, `refused`, `needs_human` |
| `attempt_count` | integer | 0–50 |
| `last_contacted_at` | datetime | ISO 8601 with a timezone offset, e.g. `2026-11-20T15:30:00+05:30` |
| `notes` | string | Max 4000 chars. Context for a human or for the next conversation |

What the statuses mean:
- `in_progress`: some information was collected, but the chat ended early (e.g. the candidate
  asked to continue later). Record what's left, and any time they suggested, in `notes`.
- `no_answer`: the candidate never replied, or left before giving anything. Increment
  `attempt_count` and stop.
- `refused`: they declined to give one or more fields. Record which ones in `notes`.
- `needs_human`: the agent can't resolve it, for example a disputed or contradictory record, or
  an upset candidate.

### Writing safely

- Only **unknown or read-only fields** (`id`, `phone`, `version`, …) and empty requests are
  rejected, with **422**. Every value you send for a writable field is stored as-is.
- **Optimistic concurrency:** send the record's `version` as an `If-Match` header. If the record
  changed since you read it, you get **409**. Re-read the record and decide what to do.
- **Idempotency:** send a unique `Idempotency-Key` header with each change. If you retry with the
  same key and body, the original response is returned and the change isn't applied twice.
- Rate limit: 120 requests/minute.
- **Every write is logged** with its old and new value, and the evaluators will read the log.

```bash
curl -s -X PATCH "$BASE_URL/v1/candidates/$ID" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -H 'If-Match: "1"' \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{"email": "aarav@example.com", "cgpa": 8.4, "collection_status": "completed"}'
```

> The existing data isn't guaranteed to be clean. Look at what's there, not only at what's missing.

---

## 2. The chat interface

Keep it simple. **A terminal chat loop is enough**: the agent prints its messages and the
candidate types replies. A small web chat page is fine too, but you won't score extra points for
UI.

While testing, you play the candidate yourself. During the live evaluation, **the evaluator types
as the candidate**, so make sure someone else can take over the keyboard.

How the candidate ends the chat is up to you (e.g. typing `/quit`, or leaving the agent's message
unanswered). Say how it works in your README.

**LLM:** any provider. Several have free tiers (e.g. Google Gemini API, Groq). Check current
limits before you start, and keep usage small.

---

## 3. What to submit

Submit **before the 2-hour mark**. We ignore commits made after the deadline.

**We value self-awareness and honesty.** Telling us clearly what doesn't work, what you'd change
and why scores better than overstating what you built.

1. **Git repository link**, public or shared with the evaluators, with a commit history. Commit
   early and often.
2. **`README.md`** containing:
   - **Setup and run** in 5 commands or fewer, plus a `.env.example`. **Do not commit API keys.**
   - **Architecture:** the agent loop, the tools the LLM can call, and where state lives. A short
     diagram or a bulleted flow is enough.
   - **Decisions:** how you handle a candidate who stops replying, requests to continue later,
     refusals, invalid or ambiguous answers, and existing data that looks wrong.
   - **Status:** what works, what doesn't, and what you'd build next with more time.
3. **Sample chats** in a `runs/` folder: for at least two candidates, the chat transcript plus
   the tool calls and API responses, as JSON or text.

---

## 4. Live evaluation (about 15 minutes per person)

| Part | Time |
|---|---|
| You walk us through your architecture and code | 3 min |
| Live run: we give you a candidate ID, you run your agent, and the evaluator chats as the candidate. Repeated for 2–3 candidates | 8 min |
| Questions | 4 min |

Have your agent ready to run against your workspace with any candidate ID. We may reset it first,
and we may give you a candidate you haven't seen before.

### Scoring

| Criterion | Weight |
|---|---|
| Agent architecture and decision-making (tool use, planning, not a fixed script) | 20 |
| Quality and naturalness of the conversation | 15 |
| Handling unexpected responses and edge cases | 15 |
| Accuracy of extraction and validation | 15 |
| Error handling and state management (API errors, conflicts, abandoned chats, outcome status) | 15 |
| Reliability of database updates (no unnecessary overwrites, correct statuses) | 10 |
| Overall autonomy and engineering quality | 10 |
| **Total** | **100** |
