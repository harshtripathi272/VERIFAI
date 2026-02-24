"""
VERIFAI Mobile Demo — Simple HTTP Server

Serves the mobile demo as static files and provides the correct
CORS headers needed for Transformers.js (SharedArrayBuffer requires
cross-origin isolation headers).

Usage:
    python serve.py

Then open http://<your-ip>:8080 on your phone.
"""

import http.server
import socket
import sys
import os


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS + cross-origin isolation headers."""

    def end_headers(self):
        # Required for SharedArrayBuffer (ONNX Runtime multithreading)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()


def get_local_ip():
    """Get the LAN IP address so you can access from your phone."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

    # Change to the mobile_demo directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    local_ip = get_local_ip()

    print()
    print("=" * 60)
    print("  VERIFAI Mobile Demo Server")
    print("=" * 60)
    print()
    print(f"  Local:    http://localhost:{port}")
    print(f"  Network:  http://{local_ip}:{port}")
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print(f"  │  Open this on your phone:                   │")
    print(f"  │  http://{local_ip}:{port:<24}│")
    print("  │  (both devices must be on the same Wi-Fi)   │")
    print("  └─────────────────────────────────────────────┘")
    print()
    print("  Press Ctrl+C to stop the server")
    print()

    server = http.server.HTTPServer(('0.0.0.0', port), CORSHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == '__main__':
    main()
