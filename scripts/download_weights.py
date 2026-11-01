"""Download the two released FP16 weights from GitHub Release assets.

Usage (from the repository root):
    python scripts/download_weights.py

The downloader supports HTTP range resume. If a network drops, run it again;
it continues from the existing .part file instead of restarting.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/HitlightHU/doudizhu-ai/releases/download/v1.0.0"
FILES = {
    "best_model.pt": ("C7DDCE6900E7E2243704F9362727E9BDA225C0F316C72D2588308C2B71B7E5E0", 11229155),
    "best_model_noshuffle.pt": ("5F702526994AD6A3D61599D45421D2A1EB9EAF02BB9CB995834E37D5D18361ED", 11230022),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def curl_fetch(url: str, tmp: Path, expected_size: int) -> bool:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        return False
    cmd = [
        curl, "-L", "--fail", "--silent", "--show-error",
        "--continue-at", "-", "--retry", "8", "--retry-all-errors",
        "--retry-delay", "2", "--connect-timeout", "30", "--max-time", "180",
        "-o", str(tmp), url,
    ]
    if os.name == "nt":
        # Windows Schannel: the VPN certificate chain may have no reachable CRL.
        cmd.insert(3, "--ssl-no-revoke")
    for attempt in range(1, 6):
        proc = subprocess.run(cmd)
        size = tmp.stat().st_size if tmp.exists() else 0
        if proc.returncode == 0 and size == expected_size:
            return True
        if size >= expected_size:
            return False
        print(f"[retry] curl attempt {attempt} stopped at {size/1024/1024:.1f} MiB", flush=True)
    return False


def urllib_fetch(url: str, tmp: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=180) as r, tmp.open("ab") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[info] urllib failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (expected_hash, expected_size) in FILES.items():
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size == expected_size and sha256(dst) == expected_hash:
            print(f"[skip] {name} already valid")
            continue
        url = f"{BASE}/{name}"
        tmp = dst.with_suffix(dst.suffix + ".part")
        if tmp.exists() and tmp.stat().st_size >= expected_size:
            tmp.unlink()
        print(f"[get ] {name} <- {url}", flush=True)

        # Never append from byte 0 to an existing partial file.
        if tmp.exists() and tmp.stat().st_size > 0:
            ok = curl_fetch(url, tmp, expected_size)
        else:
            ok = curl_fetch(url, tmp, expected_size) or urllib_fetch(url, tmp)
        if not ok or not tmp.exists():
            print(f"[fail] {name}: incomplete download; rerun to resume", file=sys.stderr)
            return 1
        size = tmp.stat().st_size
        digest = sha256(tmp)
        if size != expected_size or digest != expected_hash:
            tmp.unlink(missing_ok=True)
            print(f"[fail] {name}: size/hash mismatch: size={size} hash={digest}", file=sys.stderr)
            return 1
        tmp.replace(dst)
        print(f"[ok  ] {name} ({size/1024/1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
