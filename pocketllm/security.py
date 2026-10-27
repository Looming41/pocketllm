"""Air-gap mode for organisations that must not leak data (defence, government,
regulated companies).

What it does
  * policy file ``airgap.json`` on the drive turns cloud features off for good:
    no Meta Model API, no model downloads, UI bound to 127.0.0.1 only
  * network detection without sending a packet (routing table / interfaces),
    because a probe to a public IP is itself a policy violation on a closed network
  * signed manifest (SHA-256 of every file + OpenSSL signature) so the drive can be
    checked for tampering before anything executes from it
  * drive encryption status (FileVault/APFS, LUKS, BitLocker) shown to the user
  * audit log with metadata only (time, model, token counts, prompt hash),
    never the prompt text itself, unless the policy explicitly allows it

Signing uses the ``openssl`` CLI (present on macOS and Linux) so PocketLLM stays
stdlib-only. Keys never live on the drive: the provisioning station holds the
private key, the drive carries only the public key.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time

DEFAULT_POLICY = {
    "airgap": False,            # true: every cloud feature is disabled
    "require_offline": False,   # true: refuse to start while a network route exists
    "verify_manifest": "quick", # off | quick (code + binaries + model headers) | full
    "audit_log": True,
    "audit_prompt_text": False, # store prompt text in the audit log (usually NO on classified systems)
    "allow_downloads": True,
}

MANIFEST = "manifest.json"
SIGNATURE = "manifest.sig"
PUBKEY = "pocketllm_pub.pem"
SAMPLE = 4 << 20   # quick mode: hash the first and last 4 MiB of large files


def load_policy(root: str) -> dict:
    pol = dict(DEFAULT_POLICY)
    path = os.path.join(root, "airgap.json")
    if os.path.exists(path):
        with open(path) as f:
            pol.update(json.load(f))
    if os.environ.get("POCKETLLM_AIRGAP") == "1":
        pol["airgap"] = True
    if pol["airgap"]:
        pol["allow_downloads"] = False
    return pol


# ------------------------------------------------------------------ network
def network_routes() -> dict:
    """Detect connectivity without sending packets: default route and up interfaces."""
    system = platform.system()
    info = {"default_route": None, "interfaces": [], "method": None}
    if system == "Linux":
        info["method"] = "/proc/net/route"
        try:
            for line in open("/proc/net/route").read().splitlines()[1:]:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000":
                    info["default_route"] = parts[0]
        except OSError:
            pass
        try:
            for ifc in os.listdir("/sys/class/net"):
                if ifc == "lo":
                    continue
                state = open(f"/sys/class/net/{ifc}/operstate").read().strip()
                if state in ("up", "unknown"):
                    info["interfaces"].append(ifc)
        except OSError:
            pass
    elif system == "Darwin":
        info["method"] = "route -n get default"
        out = _run(["route", "-n", "get", "default"])
        for line in out.splitlines():
            if "interface:" in line:
                info["default_route"] = line.split(":", 1)[1].strip()
        ifs = _run(["ifconfig", "-l", "-u"]).split()
        info["interfaces"] = [i for i in ifs if not i.startswith(("lo", "gif", "stf", "anpi", "ap", "awdl", "llw", "utun", "bridge"))]
    elif system == "Windows":
        info["method"] = "Get-NetRoute"
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-NetRoute -DestinationPrefix 0.0.0.0/0 -ErrorAction SilentlyContinue | Measure-Object).Count"])
        info["default_route"] = "present" if out.strip() not in ("", "0") else None
    info["online_possible"] = bool(info["default_route"])
    return info


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------------ encryption
def encryption_status(path: str) -> dict:
    system = platform.system()
    st = {"encrypted": None, "detail": None}
    if system == "Darwin":
        out = _run(["diskutil", "info", path])
        for line in out.splitlines():
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k in ("FileVault", "Encrypted"):
                st["encrypted"] = v.lower().startswith("yes")
                st["detail"] = f"{k}: {v}"
    elif system == "Linux":
        out = _run(["findmnt", "-n", "-o", "SOURCE", "--target", path]).strip()
        if out:
            dev = os.path.realpath(out)
            tree = _run(["lsblk", "-s", "-n", "-o", "TYPE", dev])
            st["encrypted"] = "crypt" in tree.split()
            st["detail"] = f"{dev}: {' <- '.join(tree.split())}"
    elif system == "Windows":
        drive = os.path.splitdrive(os.path.abspath(path))[0]
        out = _run(["manage-bde", "-status", drive])
        if "Protection On" in out:
            st["encrypted"], st["detail"] = True, "BitLocker protection on"
        elif out:
            st["encrypted"], st["detail"] = False, "BitLocker off"
    return st


# ------------------------------------------------------------------ manifest
def _hash_file(path: str, quick: bool) -> str:
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if quick and size > 2 * SAMPLE:
            h.update(f.read(SAMPLE))
            f.seek(size - SAMPLE)
            h.update(f.read(SAMPLE))
            h.update(str(size).encode())
            return "quick:" + h.hexdigest()
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _walk(root: str):
    skip_dirs = {"data", "__pycache__", ".git", ".build"}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip_dirs]
        for fn in sorted(fns):
            if fn in (MANIFEST, SIGNATURE) or fn.endswith(".pyc"):
                continue
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            # manifest paths always use "/" so a drive sealed on one OS verifies on another
            yield os.path.relpath(p, root).replace(os.sep, "/")


def build_manifest(root: str, progress=print) -> dict:
    files = {}
    for rel in _walk(root):
        p = os.path.join(root, *rel.split("/"))
        files[rel] = {"size": os.path.getsize(p), "sha256": _hash_file(p, quick=False),
                      "quick": _hash_file(p, quick=True)}
        progress(f"  hashed {rel}")
    return {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": files}


def seal(root: str, private_key: str, progress=print) -> str:
    """Write manifest.json, sign it with the private key, copy the public key next to it."""
    if not shutil.which("openssl"):
        raise RuntimeError("openssl not found")
    m = build_manifest(root, progress)
    mpath = os.path.join(root, MANIFEST)
    with open(mpath, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    sig = os.path.join(root, SIGNATURE)
    subprocess.run(["openssl", "dgst", "-sha256", "-sign", private_key, "-out", sig, mpath], check=True)
    pub = subprocess.run(["openssl", "pkey", "-in", private_key, "-pubout"], check=True, capture_output=True).stdout
    with open(os.path.join(root, PUBKEY), "wb") as f:
        f.write(pub)
    return mpath


def verify(root: str, mode: str = "quick", trusted_pubkey: str | None = None) -> dict:
    """Check signature, then file hashes. mode: quick (sampled large files) or full."""
    res = {"ok": False, "signature": None, "missing": [], "modified": [], "unexpected": [], "checked": 0}
    mpath, sig = os.path.join(root, MANIFEST), os.path.join(root, SIGNATURE)
    if not os.path.exists(mpath):
        res["error"] = "no manifest on drive (run `pocketllm seal` on the provisioning station)"
        return res
    pub = trusted_pubkey or os.path.join(root, PUBKEY)
    if os.path.exists(sig) and not shutil.which("openssl"):
        # a signed drive whose signature cannot be checked must not pass (e.g. Windows without OpenSSL)
        res["signature"] = False
        res["error"] = "cannot verify the drive signature: openssl not found (install it, e.g. with Git for Windows)"
        return res
    if os.path.exists(sig) and os.path.exists(pub):
        r = subprocess.run(["openssl", "dgst", "-sha256", "-verify", pub, "-signature", sig, mpath],
                           capture_output=True, text=True)
        res["signature"] = r.returncode == 0
        if trusted_pubkey is None:
            res["signature_note"] = "verified with the key on the drive; pass the organisation's key for real trust"
    if res["signature"] is False:
        # never trust (or even parse) a manifest whose signature does not match
        res["error"] = "manifest signature is invalid: the drive was modified after sealing"
        return res
    try:
        with open(mpath) as f:
            m = json.load(f)
    except ValueError:
        res["error"] = "manifest is corrupt"
        return res
    present = set(_walk(root))
    for rel, meta in m["files"].items():
        p = os.path.join(root, *rel.split("/"))
        if rel not in present:
            res["missing"].append(rel)
            continue
        if os.path.getsize(p) != meta["size"]:
            res["modified"].append(rel)
            continue
        quick = mode == "quick"
        h = _hash_file(p, quick)
        if h != (meta["quick"] if quick else meta["sha256"]):
            res["modified"].append(rel)
        res["checked"] += 1
    res["unexpected"] = sorted(present - set(m["files"]) - {PUBKEY})
    res["ok"] = bool(res["signature"] is not False and not res["missing"] and not res["modified"]
                     and not res["unexpected"])
    return res


# ------------------------------------------------------------------ audit
class AuditLog:
    def __init__(self, path: str, policy: dict):
        self.path = path
        self.enabled = policy.get("audit_log", True)
        self.text = policy.get("audit_prompt_text", False)

    def write(self, event: str, **fields):
        if not self.enabled:
            return
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event}
        for k, v in fields.items():
            if k == "prompt":
                rec["prompt_sha256"] = hashlib.sha256(v.encode()).hexdigest()[:16]
                rec["prompt_chars"] = len(v)
                if self.text:
                    rec["prompt"] = v
            else:
                rec[k] = v
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
