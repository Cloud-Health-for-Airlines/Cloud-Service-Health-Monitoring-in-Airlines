"""HTTP-to-TCP integration boundary between cloud services and the legacy core."""

import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path != "/legacy":
            self.send_error(404, "use /legacy?caller=<service>")
            return
        caller = parse_qs(urlparse(self.path).query).get("caller", ["unknown"])[0]
        try:
            with socket.create_connection(("legacy-core", 9090), timeout=2) as connection:
                connection.sendall(f"LOOKUP {caller}\n".encode())
                legacy_reply = connection.recv(1024)
        except OSError as error:
            self.send_error(502, f"legacy-core error: {error}")
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"boundary-gateway: " + legacy_reply)

    def log_message(self, format, *args):
        pass


HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
