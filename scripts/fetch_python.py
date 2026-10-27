"""Put a portable CPython on the drive, so PocketLLM runs on computers without Python
(common on locked-down, air-gapped machines). Run this on the online provisioning station.

Uses python-build-standalone "install_only" builds (astral-sh, MPL-2.0 / PSF): a relocatable
interpreter that runs from any directory, including a USB drive. Every archive is checked
against the release's SHA256SUMS before it is unpacked.

usage: python3 scripts/fetch_python.py DEST/PocketLLM [--platforms linux-x64 macos-arm64 windows-x64]
       [--version 3.12] [--release TAG]
The launchers look for DEST/PocketLLM/python/<platform>/ first.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request

API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases"
TRIPLES = {
    "linux-x64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
    "macos-arm64": "aarch64-apple-darwin",
    "macos-x64": "x86_64-apple-darwin",
    "windows-x64": "x86_64-pc-windows-msvc",
}


def get(url, binary=False):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "pocketllm-provisioning"})
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok and url.startswith("https://api.github.com"):
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    return data if binary else data.decode()


def pick(assets, triple, version):
    pat = re.compile(rf"^cpython-{re.escape(version)}\.\d+\+\d+-{triple}-install_only\.tar\.gz$")
    names = sorted((a for a in assets if pat.match(a["name"])), key=lambda a: a["name"])
    return names[-1] if names else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--platforms", nargs="+", default=["linux-x64", "macos-arm64", "windows-x64"])
    ap.add_argument("--version", default="3.12")
    ap.add_argument("--release", help="release tag (default: latest)")
    a = ap.parse_args()
    rel = json.loads(get(f"{API}/tags/{a.release}" if a.release else f"{API}/latest"))
    assets = rel["assets"]
    sums_asset = next((x for x in assets if x["name"] == "SHA256SUMS"), None)
    if not sums_asset:
        sys.exit("release has no SHA256SUMS; refusing to install unverified binaries")
    sums = {}
    for line in get(sums_asset["browser_download_url"]).splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    for plat in a.platforms:
        asset = pick(assets, TRIPLES[plat], a.version)
        if not asset:
            print(f"{plat}: no CPython {a.version} build in release {rel['tag_name']}")
            continue
        blob = get(asset["browser_download_url"], binary=True)
        digest = hashlib.sha256(blob).hexdigest()
        if sums.get(asset["name"]) != digest:
            sys.exit(f"{asset['name']}: SHA-256 mismatch, not installed")
        out = os.path.join(a.dest, "python", plat)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "py.tar.gz")
            with open(arc, "wb") as f:
                f.write(blob)
            with tarfile.open(arc) as t:
                if hasattr(tarfile, "data_filter"):
                    t.extractall(tmp, filter="data")
                else:
                    for m in t.getmembers():  # no absolute paths or parent escapes
                        if m.name.startswith("/") or ".." in m.name.split("/"):
                            sys.exit(f"unsafe path in archive: {m.name}")
                    t.extractall(tmp)
            shutil.rmtree(out, ignore_errors=True)
            shutil.move(os.path.join(tmp, "python"), out)
        print(f"{plat}: {asset['name']} ({len(blob) / 1e6:.0f} MB, sha256 ok) -> {out}")


if __name__ == "__main__":
    main()
