from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class ServerBehavior:
    """Controls how the fake publisher server responds."""

    files: dict[str, bytes] = field(default_factory=dict)
    honor_range: bool = True
    # Close the connection after sending this many body bytes on the next N responses.
    truncate_after: int | None = None
    truncate_times: int = 0
    requests: list[tuple[str, str | None]] = field(default_factory=list)
    user_agents: list[str | None] = field(default_factory=list)


class _Handler(BaseHTTPRequestHandler):
    behavior: ServerBehavior

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        behavior = self.behavior
        path, _, _ = self.path.partition("?")
        range_header = self.headers.get("Range")
        behavior.requests.append((path, range_header))
        behavior.user_agents.append(self.headers.get("User-Agent"))

        if path.startswith("/redirect/"):
            # Mimic CaltechDATA: the record URL redirects to a storage URL.
            self.send_response(302)
            self.send_header("Location", "/files/" + path.removeprefix("/redirect/"))
            self.end_headers()
            return

        key = path.removeprefix("/files/")
        body = behavior.files.get(key)
        if body is None:
            self.send_error(404)
            return

        start = 0
        status = 200
        if range_header and behavior.honor_range:
            start = int(range_header.removeprefix("bytes=").split("-")[0])
            status = 206
        payload = body[start:]
        declared = len(payload)
        if behavior.truncate_after is not None and behavior.truncate_times > 0:
            behavior.truncate_times -= 1
            payload = payload[: behavior.truncate_after]

        self.send_response(status)
        self.send_header("Content-Length", str(declared))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.end_headers()
        self.wfile.write(payload)
        if len(payload) < declared:
            self.close_connection = True


@dataclass
class FakeServer:
    base_url: str
    behavior: ServerBehavior


@pytest.fixture
def fake_server() -> Iterator[FakeServer]:
    behavior = ServerBehavior()
    handler = type("Handler", (_Handler,), {"behavior": behavior})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    try:
        yield FakeServer(f"http://127.0.0.1:{server.server_address[1]}", behavior)
    finally:
        server.shutdown()
        server.server_close()
