"""Download a public DesInventar standard database export ZIP and record provenance."""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_BASE = "https://www.desinventar.net/DesInventar/download/DI_export_{code}.zip"

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description="Download a DesInventar standard export")
    p.add_argument("--country-code", required=True, help="DesInventar code such as npl")
    p.add_argument("--output-dir", default="data/raw/desinventar")
    p.add_argument("--url", default="")
    a = p.parse_args()
    code = a.country_code.strip().lower()
    url = a.url.strip() or DEFAULT_BASE.format(code=code)
    outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"DI_export_{code}.zip"
    req = Request(url, headers={"User-Agent": "RAKSHA-AI-research-ingestor/1.0"})
    try:
        with urlopen(req, timeout=60) as resp, open(out, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as exc:
        print(json.dumps({"error": "download_failed", "message": str(exc), "url": url}, indent=2))
        return 2
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "country_code": code,
        "source_name": "DesInventar",
        "source_url": url,
        "artifact": str(out),
        "sha256": sha256_file(out),
    }
    (outdir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
