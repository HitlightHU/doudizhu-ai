"""Download the two released weights without Git LFS.

Usage (from the repository root):
    python scripts/download_weights.py

If Git LFS is blocked by a proxy, clone with:
    $env:GIT_LFS_SKIP_SMUDGE=1
    git clone https://github.com/HitlightHU/doudizhu-ai.git
then run this script.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://media.githubusercontent.com/media/HitlightHU/doudizhu-ai/main/checkpoints"
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
        print(f"[get ] {name} <- {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as first_error:
            # Some VPN/proxy MITM setups break Python's OpenSSL but not Windows Schannel.
            curl = shutil.which("curl.exe") or shutil.which("curl")
            if not curl:
                tmp.unlink(missing_ok=True)
                print(f"[fail] {name}: {first_error}", file=sys.stderr)
                return 1
            print(f"[info] urllib failed ({first_error}); retry with curl")
            tmp.unlink(missing_ok=True)
            proc = subprocess.run([
                curl, "-L", "--fail", "--ssl-no-revoke",
                "--retry", "3", "--retry-delay", "2",
                "--connect-timeout", "30", "--max-time", "300",
                "-o", str(tmp), url,
            ])
            if proc.returncode != 0:
                tmp.unlink(missing_ok=True)
                print(f"[fail] {name}: curl exit {proc.returncode}", file=sys.stderr)
                return 1
        try:
            size = tmp.stat().st_size
            digest = sha256(tmp)
            if size != expected_size or digest != expected_hash:
                raise RuntimeError(f"size/hash mismatch: size={size} hash={digest}")
            tmp.replace(dst)
            print(f"[ok  ] {name} ({size/1024/1024:.1f} MiB)")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"[fail] {name}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
