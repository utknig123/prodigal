"""Settings from environment / .env (stdlib only)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API = "https://candidate-db-api.service.external.usea2.aws.prodigaltech.com"


def load_env(path=None):
    p = Path(path) if path else ROOT / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_clients():
    from .db import CandidateDB
    from .llm import LLM
    load_env()
    db = CandidateDB(os.environ.get("CANDIDATE_API_URL", DEFAULT_API), os.environ.get("CANDIDATE_API_KEY", ""))
    return db, LLM()
