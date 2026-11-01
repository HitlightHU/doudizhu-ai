"""Download released weights from GitHub Release assets.

Usage (from the repository root):
    python scripts/download_weights.py                 # default FP32
    python scripts/download_weights.py --precision fp16
    python scripts/download_weights.py --precision both

FP32 keeps training precision (~22.4MB each). FP16 is ~11.2MB each and is
loaded into FP32 for inference. Downloads support HTTP range resume; rerun
the script after a network drop and it continues from the .part file.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/HitlightHU/doudizhu-ai/releases/download/v1.0.0"

FP32 = {
    "best_model.pt": ("best_model.pt", "258E793FB84C09F92246D8696B5DEBBF5CDA87C7C445ADDED995344318123E49", 22450700),
    "best_model_noshuffle.pt": ("best_model_noshuffle.pt", "BD91D6924EABC5DCC3A11C2AF75EB69113E4217BFFBB5E636DA2A11EA7A6AE34", 22451127),
}
FP16 = {
    "best_model.pt": ("best_model_fp16.pt", "E8E98EF03EEC7A72428133263F9B400D397992E47814D94C55D623425BBEBC01", 11229126),
    "best_model_noshuffle.pt": ("best_model_noshuffle_fp16.pt", "88F4A09586DE85B6661625D5FB170F16D156DF68752DE8D82FC3261E2749E13B", 11230376),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def plan(precision: str):
    items = []
    if precision in ("fp32", "both"):
        for dest, (asset, digest, size) in FP32.items():
            items.append((dest, asset, digest, size))
    if precision in ("fp16", "both"):
        for dest, (asset, digest, size) in FP16.items():
            out_name = dest if precision == "fp16" else dest.replace(".pt", "_fp16.pt")
            items.append((out_name, asset, digest, size))
    return items


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
        # Windows Schannel: VPN certificate chain may have no reachable CRL.
        cmd.insert(3, "--ssl-no-revoke")
    for attempt in range(1, 8):
        proc = subprocess.run(cmd)
        size = tmp.stat().st_size if tmp.exists() else 0
        if proc.returncode == 0 and size == expected_size:
            return True
        if size >= expected_size:
            return False
        print(f"[retry] {tmp.name}: attempt {attempt} stopped at {size/1024/1024:.1f} MiB", flush=True)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "fp16", "both"], default="fp32",
                    help="fp32 = exact/default; fp16 = half-size; both = download all")
    args = ap.parse_args()
    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dest, asset, expected_hash, expected_size in plan(args.precision):
        dst = out_dir / dest
        if dst.exists() and dst.stat().st_size == expected_size and sha256(dst) == expected_hash:
            print(f"[skip] {dest} already valid")
            continue
        url = f"{BASE}/{asset}"
        tmp = dst.with_suffix(dst.suffix + ".part")
        if tmp.exists() and tmp.stat().st_size >= expected_size:
            tmp.unlink()
        print(f"[get ] {dest} <- {url}", flush=True)

        if tmp.exists() and tmp.stat().st_size > 0:
            ok = curl_fetch(url, tmp, expected_size)
        else:
            ok = curl_fetch(url, tmp, expected_size) or urllib_fetch(url, tmp)
        if not ok or not tmp.exists():
            print(f"[fail] {dest}: incomplete; rerun to resume", file=sys.stderr)
            return 1
        size = tmp.stat().st_size
        digest = sha256(tmp)
        if size != expected_size or digest != expected_hash:
            tmp.unlink(missing_ok=True)
            print(f"[fail] {dest}: size/hash mismatch: size={size} hash={digest}", file=sys.stderr)
            return 1
        tmp.replace(dst)
        print(f"[ok  ] {dest} ({size/1024/1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
