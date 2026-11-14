# Project prep: what works, what I would do with more time

## Done for the 2-hour submission shape
- One-candidate CLI agent (`python agent.py --candidate-id c00x`) and a small web UI.
- Deterministic audit before chat; no conversation when nothing is needed.
- Tool-using LLM loop with a hard harness: verbatim evidence, validators, identity gate, no silent overwrites, verified `completed`.
- Safe PATCH path: `If-Match`, idempotency, 409 re-read, retries.
- Offline unit tests + a live LLM eval harness against an in-memory DB.
- Groq wired as the default provider. Model is whatever this key can actually list (currently `openai/gpt-oss-20b`).

## Blocked until you send / configure
- **Candidate DB key** (`CANDIDATE_API_KEY`): `ASSIGNMENT.md` does not contain one. Without it, live listing, PATCH and sample `runs/` against the real workspace cannot be finished.
- **AWS deploy**: the AWS CLI needs an **IAM access key + secret**, not the console email/password. Password was never stored in this project. Configure with `aws configure`, then `python scripts/aws_tool.py doctor`.

## With more time (why skipped now)
| Next step | Why it matters | Why not in this pass |
|---|---|---|
| DynamoDB sessions + EventBridge sweeper | Abandoned web chats that never send a leave beacon stay `in_progress` forever | Extra infra and IAM for a 2-hour assignment; CLI already closes on idle/`/quit` |
| Cognito / access code on the Function URL | Public URL can drive the workspace | Demo scope; secrets stay in Lambda env |
| Secrets Manager for keys | Safer than env params | Deploy script would get longer |
| LLM-simulated adversarial candidates in CI | Catch regressions without a human | Needs stable quota and a flaky-test policy |
| Scheduler for `in_progress` follow-ups | Assignment says the agent records and stops | Explicitly out of scope |
| Stronger CGPA scale handling per college | Real forms mix 10-point and percentage | Risk of inventing conversions; we refuse instead |
| Multilingual / WhatsApp channel | Production WhatsApp flow | Assignment says local chat is enough |
| Broader live `runs/` on every seed candidate | Better evaluator evidence | Needs the personal candidate API key and time to role-play |

## Security note for this machine
A console password was pasted into chat once. Treat it as exposed: change it in AWS, turn on MFA, and create a least-privilege IAM user with access keys for CLI only. Do not paste passwords into the repo or into chat again.
