#!/usr/bin/env python3
"""
Proxy server for Codex CLI to convert developer role to system role
Workaround for: https://github.com/your-repo/issues
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

# Configuration
PROXY_PORT = 8765
TARGET_API_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"

class RoleConversionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

    def do_POST(self):
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            # Parse JSON
            data = json.loads(body.decode('utf-8'))

            # Convert developer role to system role
            if 'messages' in data:
                for message in data['messages']:
                    if message.get('role') == 'developer':
                        message['role'] = 'system'

            # Forward to target API
            req = urllib.request.Request(
                TARGET_API_URL,
                data=json.dumps(data).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': self.headers.get('Authorization', ''),
                }
            )

            response = urllib.request.urlopen(req)
            response_body = response.read()

            # Send response back to client
            self.send_response(response.status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response_body)

        except urllib.error.HTTPError as e:
            # Forward error response
            error_body = e.read().decode('utf-8')
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error_body.encode('utf-8'))

        except Exception as e:
            # Internal server error
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

def run_proxy():
    server = HTTPServer(('localhost', PROXY_PORT), RoleConversionHandler)
    print(f"Codex proxy running on http://localhost:{PROXY_PORT}")
    print(f"Forwarding to: {TARGET_API_URL}")
    print("Press Ctrl+C to stop...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == '__main__':
    run_proxy()
