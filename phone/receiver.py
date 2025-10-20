# receiver.py — minimal HTTP→CSV appender
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, csv, time

OUT = "live_health.csv"

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length).decode("utf-8") or "{}"
        try:
            data = json.loads(raw)
        except Exception:
            self.send_response(400); self.end_headers()
            self.wfile.write(b"Invalid JSON"); return

        rows = data if isinstance(data, list) else [data]
        with open(OUT, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                ts = r.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                # Write one row per metric in the payload (flexible format)
                for k, v in r.items():
                    if k in ("ts", "unit", "source"): 
                        continue
                    w.writerow([ts, k, v, r.get("unit",""), r.get("source","galaxy_watch")])

        self.send_response(200); self.end_headers()

if __name__ == "__main__":
    # Listens on all interfaces, port 8000
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
