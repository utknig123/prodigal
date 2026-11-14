"""Proof that a quote came from the candidate, not from the model."""

from __future__ import annotations

import re


def contains_quote(evidence: str, candidate_messages: list[str]) -> bool:
    """True when evidence is a contiguous span of something the candidate typed."""
    parts = _norm(evidence).split(" ")
    parts = [part for part in parts if part]
    if not parts:
        return False
    pattern = r"\s+".join(re.escape(part) for part in parts)
    if parts[0][:1].isalnum():
        pattern = r"(?<![A-Za-z0-9])" + pattern
    if parts[-1][-1:].isalnum():
        pattern = pattern + r"(?![A-Za-z0-9])"
    haystack = "\n".join(candidate_messages)
    return re.search(pattern, haystack, flags=re.IGNORECASE) is not None


def find_span(value: str, evidence: str) -> str | None:
    """Return the candidate's own spelling of value inside evidence."""
    parts = _norm(value).split(" ")
    parts = [part for part in parts if part]
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(part) for part in parts)
    if parts[0][:1].isalnum():
        pattern = r"(?<![A-Za-z0-9])" + pattern
    if parts[-1][-1:].isalnum():
        pattern = pattern + r"(?![A-Za-z0-9])"
    match = re.search(pattern, evidence, flags=re.IGNORECASE)
    if not match:
        return None
    return evidence[match.start() : match.end()].strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()
