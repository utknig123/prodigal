"""Local web server:  python -m acme_agent.local  ->  http://127.0.0.1:8080"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .web import dispatch


class H(BaseHTTPRequestHandler):
    def _go(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        code, headers, payload = dispatch(self.command, self.path, body)
        data = payload.encode()
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
    do_GET = do_POST = do_OPTIONS = do_HEAD = _go

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("http://127.0.0.1:8080")
    ThreadingHTTPServer(("127.0.0.1", 8080), H).serve_forever()
