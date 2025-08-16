import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length).decode() if length else ''
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if self.path.endswith('/predict') or self.path.endswith('/predict/'): 
            text = data.get('text', '')
            # naive scoring: likes proportional to number of words
            likes = float(max(0.5, min(200.0, len(text.split()) * 2)))
            rts = float(max(0.1, min(50.0, len(text.split()) * 0.3)))
            reps = float(max(0.0, min(20.0, len(text.split()) * 0.1)))
            resp = {'likes': likes, 'retweets': rts, 'replies': reps}
            self._send(200, resp)
            return

        if self.path.endswith('/elo') or self.path.endswith('/elo/'):
            texts = data.get('texts', [])
            # score by length (longer => higher elo) for demonstration
            ranked = sorted(texts, key=lambda t: len(t), reverse=True)
            out = []
            base = 1500
            for i, t in enumerate(ranked):
                out.append({'text': t, 'elo_score': base + (len(ranked) - i) * 10})
            self._send(200, out)
            return

        # default
        self._send(404, {'error': 'unknown endpoint'})

if __name__ == '__main__':
    server = HTTPServer(('localhost', 9999), Handler)
    print('Mock scorer listening on http://localhost:9999 (endpoints: /predict, /elo)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print('Mock scorer stopped')
