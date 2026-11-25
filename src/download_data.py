#!/usr/bin/env python
"""Download & extract Fongbe ASR audio archives from the public GCS bucket.

Bucket URL (posted in Zindi Chat):
    https://storage.googleapis.com/fongbe/

Usage:
    # List all files available in the bucket
    python src/download_data.py --list

    # Download & extract 1 reliable archive (quick start / benchmark)
    python src/download_data.py --out-dir /tmp/fongbe --match audio_train_reliable --limit 1

    # Download & extract all reliable + religious + test archives
    python src/download_data.py --out-dir /tmp/fongbe --match audio_train_reliable audio_train_religious audio_test
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET_URL = "https://storage.googleapis.com/fongbe"


def list_bucket(bucket_url: str = BUCKET_URL) -> list[tuple[str, int]]:
    """Return list of (key, size_bytes) from public GCS XML listing (handles pagination)."""
    items: list[tuple[str, int]] = []
    marker = ""
    while True:
        url = f"{bucket_url}/" + (f"?marker={urllib.parse.quote(marker)}" if marker else "")
        with urllib.request.urlopen(url, timeout=60) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        is_truncated = False
        last_key = ""
        for elem in root:
            tag = elem.tag.split("}")[-1]
            if tag == "IsTruncated" and (elem.text or "").lower() == "true":
                is_truncated = True
            elif tag == "Contents":
                key, size = "", 0
                for child in elem:
                    ctag = child.tag.split("}")[-1]
                    if ctag == "Key":
                        key = child.text or ""
                    elif ctag == "Size":
                        size = int(child.text or 0)
                if key:
                    items.append((key, size))
                    last_key = key
        if not is_truncated or not last_key:
            break
        marker = last_key
    return items


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(key: str, dst: Path, bucket_url: str = BUCKET_URL) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{bucket_url}/{urllib.parse.quote(key)}"
    print(f"Downloading {url} -> {dst} ...", flush=True)
    urllib.request.urlretrieve(url, str(dst))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Fongbe audio archives from public GCS bucket.")
    ap.add_argument("--bucket-url", default=BUCKET_URL)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/fongbe"))
    ap.add_argument("--list", action="store_true", help="Only list bucket files and exit")
    ap.add_argument("--match", nargs="+", default=[], help="Substrings to match keys (e.g. audio_train_reliable)")
    ap.add_argument("--limit", type=int, default=0, help="Max number of matched archives to download (0 = all)")
    ap.add_argument("--keep-tar", action="store_true", help="Do not delete .tar after extraction")
    a = ap.parse_args()

    items = list_bucket(a.bucket_url)
    if a.list or not a.match:
        total = sum(sz for _, sz in items)
        print(f"Bucket {a.bucket_url}/ — {len(items)} files ({total / 1e9:.2f} GB total):")
        for k, sz in items:
            print(f"  {sz / 1e6:8.1f} MB   {k}")
        if a.list:
            return

    # Optional SHA256SUMS verification if available in the bucket
    expected_sha: dict[str, str] = {}
    keys_set = {k for k, _ in items}
    if "SHA256SUMS" in keys_set:
        sha_path = a.out_dir / "SHA256SUMS"
        if not sha_path.exists():
            download_file("SHA256SUMS", sha_path, a.bucket_url)
        for line in sha_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                expected_sha[parts[1].lstrip("*")] = parts[0]

    selected = [(k, sz) for k, sz in items if any(m in k for m in a.match)]
    if a.limit > 0:
        selected = selected[: a.limit]
    if not selected:
        raise SystemExit(f"No bucket files matched {a.match}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for key, sz in selected:
        marker = a.out_dir / f".done_{key.replace('/', '_')}"
        if marker.exists():
            print(f"Already extracted: {key}", flush=True)
            continue
        dst = a.out_dir / Path(key).name
        download_file(key, dst, a.bucket_url)
        if key in expected_sha or dst.name in expected_sha:
            exp = expected_sha.get(key) or expected_sha[dst.name]
            got = sha256_file(dst)
            if got != exp:
                dst.unlink(missing_ok=True)
                raise SystemExit(f"SHA256 mismatch for {key}: expected {exp}, got {got}")
            print(f"  SHA256 OK: {dst.name}", flush=True)
        if dst.suffix == ".tar" or dst.name.endswith(".tar.gz"):
            print(f"  Extracting {dst.name} ({sz / 1e6:.1f} MB)...", flush=True)
            with tarfile.open(dst) as tar:
                try:
                    tar.extractall(a.out_dir, filter="data")
                except TypeError:
                    tar.extractall(a.out_dir)
            if not a.keep_tar:
                dst.unlink()
        marker.touch()
        print(f"  Done: {key}", flush=True)


if __name__ == "__main__":
    main()
