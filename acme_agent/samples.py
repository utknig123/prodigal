"""Write labeled harness traces. These use a scripted model, not the live API."""

from __future__ import annotations

from pathlib import Path

from acme_agent.client import CandidateClient, Transcript
from acme_agent.llm import LLMTurn, ScriptedModel, ToolCall
from acme_agent.loop import run_collection
from acme_agent.memory import MemoryTransport, seed_record


def write_simulated_runs(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, record, steps, replies in cases():
        path = directory / filename
        transcript = Transcript(record["id"], str(path), source="simulation")
        result, _spoken = run_scripted(record, steps, replies, transcript)
        transcript.add(
            {
                "type": "note",
                "text": (
                    "Scripted model and in-memory API. This is not a live candidate-workspace run. "
                    f"Outcome {result['outcome']}."
                ),
            }
        )
        written.append(path)
    return written


def cases():
    meera = seed_record("csim1", email=None, cgpa=None)
    rohan = seed_record(
        "csim2",
        full_name="Rohan Desai",
        email="rohan@example.com",
        cgpa=None,
        phone="+919800000002",
    )
    return [
        (
            "simulated-meera-completed.json",
            meera,
            [
                LLMTurn(text="Hi Meera Iyer, this is ACME Corp about your internship form. Am I speaking with Meera Iyer?"),
                LLMTurn(tool_calls=[ToolCall("1", "confirm_identity", {"matched": True, "evidence": "yes, this is Meera Iyer"})]),
                LLMTurn(text="Thanks. What's your email?"),
                LLMTurn(
                    tool_calls=[
                        ToolCall(
                            "2",
                            "record_fields",
                            {
                                "updates": [
                                    {
                                        "field": "email",
                                        "value": "hallucinated@example.com",
                                        "evidence": "it's meera@example.com",
                                    }
                                ]
                            },
                        )
                    ]
                ),
                LLMTurn(
                    tool_calls=[
                        ToolCall(
                            "3",
                            "record_fields",
                            {
                                "updates": [
                                    {
                                        "field": "email",
                                        "value": "meera@example.com",
                                        "evidence": "it's meera@example.com",
                                    }
                                ]
                            },
                        )
                    ]
                ),
                LLMTurn(text="What's your CGPA on a 10-point scale?"),
                LLMTurn(
                    tool_calls=[
                        ToolCall(
                            "4",
                            "record_fields",
                            {"updates": [{"field": "cgpa", "value": 8.75, "evidence": "8.75"}]},
                        )
                    ]
                ),
                LLMTurn(tool_calls=[ToolCall("5", "finish", {"intent": "complete", "quote": ""})]),
            ],
            ["yes, this is Meera Iyer", "it's meera@example.com", "8.75"],
        ),
        (
            "simulated-rohan-refused.json",
            rohan,
            [
                LLMTurn(text="Hi Rohan Desai, this is ACME Corp. Am I speaking with Rohan Desai?"),
                LLMTurn(tool_calls=[ToolCall("1", "confirm_identity", {"matched": True, "evidence": "yes"})]),
                LLMTurn(text="What's your CGPA on a 10-point scale?"),
                LLMTurn(
                    tool_calls=[
                        ToolCall("2", "refuse_field", {"field": "cgpa", "evidence": "I won't share my cgpa"})
                    ]
                ),
                LLMTurn(
                    tool_calls=[
                        ToolCall("3", "finish", {"intent": "refused", "quote": "I won't share my cgpa"})
                    ]
                ),
            ],
            ["yes", "I won't share my cgpa"],
        ),
    ]


def run_scripted(record, steps, replies, transcript=None):
    client = CandidateClient(
        "https://memory.invalid",
        "test-key",
        transport=MemoryTransport([record]),
        transcript=transcript,
    )
    pending = iter(replies)
    spoken = []

    def ask(_prompt: str) -> str:
        try:
            return next(pending)
        except StopIteration as exc:
            raise AssertionError("The agent asked for another reply that the script does not have.") from exc

    result = run_collection(
        client,
        ScriptedModel(steps),
        record["id"],
        ask,
        spoken.append,
        lambda _line: None,
        transcript,
    )
    return result, spoken
