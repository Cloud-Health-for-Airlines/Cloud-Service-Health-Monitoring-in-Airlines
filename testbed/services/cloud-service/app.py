"""Synthetic cloud-native airline domain service."""

import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen


SERVICE_NAME = os.environ["SERVICE_NAME"]
SERVICE_PATH = os.environ["SERVICE_PATH"]
GATEWAY_URL = "http://boundary-gateway:8080/legacy"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != SERVICE_PATH:
            self.send_error(404, f"use {SERVICE_PATH}")
            return
        try:
            with urlopen(f"{GATEWAY_URL}?caller={SERVICE_NAME}", timeout=2) as response:
                legacy_reply = response.read()
        except Exception as error:
            self.send_error(502, f"{SERVICE_NAME} gateway error: {error}")
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"{SERVICE_NAME}: ".encode() + legacy_reply)

    def log_message(self, format, *args):
        pass


HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
