"""Deterministic TCP stub representing a legacy/mainframe core."""

import socketserver


class LegacyHandler(socketserver.StreamRequestHandler):
    def handle(self):
        request = self.rfile.readline().decode().strip()
        self.wfile.write(f"legacy-core: OK [{request}]\n".encode())


class LegacyServer(socketserver.TCPServer):
    allow_reuse_address = True


with LegacyServer(("0.0.0.0", 9090), LegacyHandler) as server:
    server.serve_forever()
