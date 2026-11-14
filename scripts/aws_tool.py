"""AWS helper (AWS CLI only):  doctor | deploy | destroy | url

Secrets are read from .env and passed to CloudFormation as NoEcho parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
sys.path.insert(0, str(ROOT))
from acme_agent.config import load_env  # noqa: E402

load_env()
REGION = os.environ.get("AWS_REGION") or "ap-south-1"
STACK = os.environ.get("STACK_NAME", "prodigal-dev")


def run(cmd, check=True):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if check and r.returncode:
        sys.exit((r.stderr or r.stdout).strip() or f"failed: {' '.join(cmd[:4])}")
    return r


def aws(*a, check=True):
    return run(["aws", *a, "--region", REGION], check)


def doctor():
    if not shutil.which("aws"):
        sys.exit("AWS CLI not found")
    r = aws("sts", "get-caller-identity", "--output", "json", check=False)
    if r.returncode:
        sys.exit("No AWS credentials. Run `aws configure` in your own terminal (region ap-south-1), then retry.")
    who = json.loads(r.stdout)
    print("account:", who["Account"], "| arn:", who["Arn"], "| region:", REGION)
    for k in ("CANDIDATE_API_KEY", "LLM_API_KEY"):
        print(f"{k}:", "set" if os.environ.get(k) else "MISSING in .env")


def package():
    BUILD.mkdir(exist_ok=True)
    tmp = BUILD / "payload.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("acme_agent", "frontend"):
            for p in (ROOT / folder).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(ROOT).as_posix())
    name = BUILD / f"prodigal-{hashlib.sha256(tmp.read_bytes()).hexdigest()[:12]}.zip"
    tmp.replace(name)
    return name


def outputs():
    r = aws("cloudformation", "describe-stacks", "--stack-name", STACK, "--query", "Stacks[0].Outputs", "--output", "json")
    return {o["OutputKey"]: o["OutputValue"] for o in json.loads(r.stdout) or []}


def deploy():
    doctor()
    for k in ("CANDIDATE_API_KEY", "LLM_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"{k} missing in .env")
    account = json.loads(aws("sts", "get-caller-identity", "--output", "json").stdout)["Account"]
    bucket = f"prodigal-artifacts-{account}-{REGION}"
    if aws("s3api", "head-bucket", "--bucket", bucket, check=False).returncode:
        cmd = ["s3api", "create-bucket", "--bucket", bucket]
        if REGION != "us-east-1":
            cmd += ["--create-bucket-configuration", f"LocationConstraint={REGION}"]
        aws(*cmd)
        aws("s3api", "put-public-access-block", "--bucket", bucket, "--public-access-block-configuration",
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true")
    z = package()
    key = f"prodigal/{z.name}"
    aws("s3", "cp", str(z), f"s3://{bucket}/{key}")
    secret = hashlib.sha256((os.environ["LLM_API_KEY"] + "session").encode()).hexdigest()
    params = {"ArtifactBucket": bucket, "ArtifactKey": key,
              "CandidateApiUrl": os.environ.get("CANDIDATE_API_URL", "https://candidate-db-api.service.external.usea2.aws.prodigaltech.com"),
              "CandidateApiKey": os.environ["CANDIDATE_API_KEY"], "LlmProvider": os.environ.get("LLM_PROVIDER", "groq"),
              "LlmModel": os.environ.get("LLM_MODEL", ""), "LlmFallbackModels": os.environ.get("LLM_FALLBACK_MODELS", ""),
              "LlmApiKey": os.environ["LLM_API_KEY"], "SessionSecret": secret}
    pf = BUILD / "params.json"
    pf.write_text(json.dumps([{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()]))
    try:
        # `deploy` does not take a params file; use create/update-stack semantic through deploy overrides
        over = [f"{k}={v}" for k, v in params.items()]
        aws("cloudformation", "deploy", "--stack-name", STACK, "--template-file", str(ROOT / "infra" / "template.yaml"),
            "--capabilities", "CAPABILITY_NAMED_IAM", "--no-fail-on-empty-changeset", "--parameter-overrides", *over)
    finally:
        pf.unlink(missing_ok=True)
    print("URL:", outputs().get("FunctionUrl"))


def destroy():
    aws("cloudformation", "delete-stack", "--stack-name", STACK)
    aws("cloudformation", "wait", "stack-delete-complete", "--stack-name", STACK)
    print("deleted")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "doctor"
    {"doctor": doctor, "deploy": deploy, "destroy": destroy, "package": lambda: print(package()),
     "url": lambda: print(outputs().get("FunctionUrl"))}[cmd]()
