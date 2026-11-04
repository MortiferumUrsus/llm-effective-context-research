"""Validate an assembled release and reproduce its numerical tables offline."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/"MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    sensitive = re.compile(r"sk-or-v1-[A-Za-z0-9]{24,}|sk-proj-[A-Za-z0-9_-]{24,}|Bearer\s+[A-Za-z0-9._-]{30,}")
    forbidden_parts = {".git", ".env", "__pycache__", "models_local", "logs"}
    for entry in manifest["files"]:
        path = ROOT/entry["path"]
        if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():
            errors.append(f"Missing or invalid path: {entry['path']}")
            continue
        if forbidden_parts.intersection(path.relative_to(ROOT).parts):
            errors.append(f"Forbidden publication path: {entry['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"Checksum mismatch: {entry['path']}")
        if path.suffix in {".md", ".py", ".txt", ".json", ".csv", ".cff"}:
            content = path.read_text(encoding="utf-8")
        elif path.name.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                content = f.read()
        else:
            continue
        if sensitive.search(content):
            errors.append(f"Credential-like content: {entry['path']}")
        if re.search(r"[\u0400-\u04ff]", content):
            errors.append(f"Non-English Cyrillic text requires review: {entry['path']}")
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
                if target.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                local = unquote(target.split("#", 1)[0])
                if local and not (path.parent/local).exists():
                    errors.append(f"Broken link in {entry['path']}: {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    # Disable sockets in the analysis subprocess to make the no-network scope explicit.
    program = (
        "import runpy,socket,sys; "
        "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled for reproduction')); "
        "sys.argv=['analysis.final_report','--out','results/reproduced']; "
        "runpy.run_module('analysis.final_report',run_name='__main__')"
    )
    subprocess.run([sys.executable, "-c", program], cwd=ROOT, check=True)
    tables = sorted((ROOT/"results/final").glob("*.csv"))
    for expected in tables:
        actual = ROOT/"results/reproduced"/expected.name
        left, right = pd.read_csv(expected, keep_default_na=False), pd.read_csv(actual, keep_default_na=False)
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                    "-p", "test_final_report.py", "-v"], cwd=ROOT, check=True)
    print(json.dumps({"manifest_files_checked": len(manifest["files"]),
        "tables_reproduced": len(tables), "model_calls": 0,
        "publication": "not performed"}, indent=2))


if __name__ == "__main__":
    main()
