"""AWS Lambda entrypoint (Function URL, payload v2)."""
import base64

from .web import dispatch


def lambda_handler(event, context):
    req = event.get("requestContext", {}).get("http", {})
    body = event.get("body") or ""
    if event.get("isBase64Encoded") and body:
        body = base64.b64decode(body).decode("utf-8", "replace")
    code, headers, payload = dispatch(req.get("method", "GET"), event.get("rawPath") or "/", body)
    return {"statusCode": code, "headers": headers, "body": payload, "isBase64Encoded": False}
