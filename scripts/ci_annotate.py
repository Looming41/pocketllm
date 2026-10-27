"""Turn the failure section of a test log into GitHub Actions error annotations, so failures
can be read through the API without downloading the full log.

usage: python scripts/ci_annotate.py test.log
"""
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
blocks = text.split("=" * 70)
fails = [b.strip() for b in blocks if b.strip().startswith(("FAIL:", "ERROR:"))]
for b in fails[:10] or [text[-3000:]]:
    print("::error::" + b.replace("%", "%25").replace("\r", "").replace("\n", "%0A")[:3500])
