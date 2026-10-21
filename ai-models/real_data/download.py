"""Download the real public datasets listed in catalog.py.

Files land in ``data/raw/<dataset_key>/`` at the repository root (git-ignored).
Downloads are resumable and skipped when the file already exists with the
size the server reports.

Examples (from the repository root)::

    python ai-models/real_data/download.py --list
    python ai-models/real_data/download.py                        # default "core" set
    python ai-models/real_data/download.py bts_ontime --param year=2023 month=1
    python ai-models/real_data/download.py alibaba_ms_2022_msmetrics --param index=0
    python ai-models/real_data/download.py rcaeval                 # RE1 Train-Ticket, 280 MB
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real_data.catalog import CATALOG, DatasetSpec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

CORE_SET = ["bts_ontime", "bts_db1b_market", "azure_functions_2019", "alibaba_ms_2022_node",
            "gcp_incidents", "gcp_incidents_archive"]
USER_AGENT = "BACCP-research-downloader/1.0 (academic project)"


def resolve_urls(spec: DatasetSpec, params: List[Dict[str, int]]) -> List[str]:
    if spec.url_template is None:
        return []
    return [spec.url_template.format(**p) for p in (params or [{}])]


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have == expected:
        print(f"    = already complete: {dest.name} ({have / 1e6:.1f} MB)")
        return dest

    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if have and expected and have < expected:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, mode) as fh:
        if mode == "ab" and resp.status != 206:  # server ignored Range; restart
            fh.seek(0)
            fh.truncate()
            have = 0
        done = have
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if expected:
                print(f"\r    ↓ {dest.name}: {done / 1e6:7.1f} / {expected / 1e6:.1f} MB", end="", flush=True)
    print()
    return dest


def download(key: str, params: List[Dict[str, Any]] | None = None) -> List[Path]:
    spec = CATALOG[key]
    urls = resolve_urls(spec, params if params is not None else spec.default_params)
    if not urls:
        print(f"[-] {key}: no direct download ({spec.approx_size}). See {spec.homepage}")
        return []
    print(f"[*] {spec.name}  ({spec.approx_size})")
    out = []
    if key == "rcaeval":
        return [_download_rcaeval(p, url) for p, url in zip(params or spec.default_params, urls)]
    if key == "gcp_incidents_archive":
        for p, url in zip(params if params is not None else spec.default_params, urls):
            dest = RAW_DIR / key / f"incidents_{p['timestamp']}.json"
            if _fetch_json_snapshot(url, dest):
                out.append(dest)
        return out
    for url in urls:
        out.append(download_file(url, RAW_DIR / key / url.rsplit("/", 1)[-1]))
    return out


RCAEVAL_RECORD_API = "https://zenodo.org/api/records/14590730"
RCAEVAL_MAX_MB = 500


def _download_rcaeval(p: Dict[str, str], url: str, max_mb: float = RCAEVAL_MAX_MB) -> Path:
    """Fetch one RCAEval archive (e.g. RE1-TT.zip) after checking its size, then unzip it."""
    import json
    import zipfile

    name = f"{p['subset']}-{p['system']}"
    dest_dir = RAW_DIR / "rcaeval"
    if (dest_dir / name).is_dir():
        print(f"    = already extracted: {dest_dir / name}")
        return dest_dir / name
    req = urllib.request.Request(RCAEVAL_RECORD_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        files = {f["key"]: f["size"] for f in json.load(resp)["files"]}
    tt = sorted((size, key) for key, size in files.items() if key.endswith(f"-{p['system']}.zip"))
    print("    archives for this system on Zenodo: " + ", ".join(f"{k} {s / 1e6:.0f} MB" for s, k in tt))
    size = files.get(f"{name}.zip")
    if size is None:
        raise SystemExit(f"{name}.zip is not in Zenodo record {RCAEVAL_RECORD_API}")
    if size / 1e6 > max_mb:
        raise SystemExit(f"{name}.zip is {size / 1e6:.0f} MB > {max_mb:.0f} MB cap; raise RCAEVAL_MAX_MB to fetch it")
    archive = download_file(url, dest_dir / f"{name}.zip")
    with zipfile.ZipFile(archive) as zf:
        root = dest_dir.resolve()
        for member in zf.namelist():  # refuse path traversal in an untrusted archive
            if not (root / member).resolve().is_relative_to(root):
                raise SystemExit(f"unsafe path in archive: {member}")
        zf.extractall(dest_dir)
    archive.unlink()
    print(f"    ✓ extracted to {dest_dir / name}")
    return dest_dir / name


def _fetch_json_snapshot(url: str, dest: Path) -> bool:
    """Fetch an archived JSON snapshot; gunzip if needed and keep it only if it parses."""
    import gzip
    import json

    if dest.exists():
        print(f"    = already have {dest.name}")
        return True
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = resp.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    try:
        n = len(json.loads(body))
    except ValueError:
        print(f"    ! {dest.name}: snapshot truncated by the archive - skipped")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    print(f"    ✓ {dest.name}: {n} incidents")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help=f"dataset keys (default: {' '.join(CORE_SET)})")
    ap.add_argument("--param", nargs="*", default=None, help="url params, e.g. year=2023 month=1")
    ap.add_argument("--list", action="store_true", help="list catalog and exit")
    args = ap.parse_args()

    if args.list:
        for spec in CATALOG.values():
            print(f"{spec.key:28s} {spec.role:17s} {spec.approx_size:40s} {spec.name}")
        return

    params = None
    if args.param:
        params = [{k: int(v) if v.isdigit() else v for k, v in (p.split("=", 1) for p in args.param)}]
    for key in args.keys or CORE_SET:
        download(key, params)


if __name__ == "__main__":
    main()
