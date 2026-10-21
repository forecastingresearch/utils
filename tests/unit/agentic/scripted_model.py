"""Serve deterministic model responses over localhost, with no provider calls."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REFUSAL = object()


class ScriptedModel:
    """Serve scripted tool calls through an OpenAI-compatible local endpoint."""

    def __init__(self, script):
        """Start a local model server and record incoming requests."""
        self.script, self.requests = list(script), []
        model = self

        class Handler(BaseHTTPRequestHandler):
            """Return scripted server-sent events for each model request."""

            def log_message(self, *a):
                """Suppress routine HTTP request logging."""
                pass

            def do_POST(self):
                """Reply with the next scripted assistant turn."""
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model.requests.append(body)
                turn = (
                    model.script[len(model.requests) - 1]
                    if len(model.requests) <= len(model.script)
                    else "done"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in model.chunks(turn, len(model.requests)):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    @staticmethod
    def chunks(turn, n):
        """Encode a tool call, refusal, or text as streaming completion chunks."""
        base = {"id": f"c{n}", "object": "chat.completion.chunk", "created": 0, "model": "scripted"}
        if turn is REFUSAL:
            delta, finish = {"role": "assistant", "content": ""}, "content_filter"
        elif isinstance(turn, str):
            delta, finish = {"role": "assistant", "content": turn}, "stop"
        else:
            name, args = turn
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{n}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
            finish = "tool_calls"
        yield dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])
        yield dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}])
        yield dict(
            base,
            choices=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def tools(self, n):
        """Return tools offered in the numbered request, starting at one."""
        return {t["function"]["name"] for t in self.requests[n - 1].get("tools") or []}

    def text(self, n, *roles):
        """Return the last matching message in the numbered request."""
        msg = [m for m in self.requests[n - 1]["messages"] if m.get("role") in roles][-1]
        content = msg["content"]
        return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)

    def last_user(self, n):
        """Return the last user message in the numbered request."""
        return self.text(n, "user")

    def tool_result(self, n):
        """Return the tool response visible in the numbered request."""
        return self.text(n, "tool")

    def close(self):
        """Stop the local server and release its listening socket."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
