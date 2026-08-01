from __future__ import annotations
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    "__pycache__", ".ipynb_checkpoints", "private_ground_truth",
    "private_labels", "locked_attack_labels", "locked_attack_truth",
    "locked_truth", "private_after_freeze",
}
FORBIDDEN_TEXT = (r"C:\Users\SADRA", "C:/Users/SADRA")
MAX_BYTES = 20 * 1024 * 1024

def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")

def main() -> None:
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    for p in files:
        if {part.lower() for part in p.parts}.intersection(FORBIDDEN_PARTS):
            fail(f"Forbidden private/cache path: {p.relative_to(ROOT)}")
        if p.stat().st_size > MAX_BYTES:
            fail(f"Unexpected large Git file: {p.relative_to(ROOT)}")
        if p.suffix.lower() in {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".cff"}:
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.resolve() != Path(__file__).resolve() and any(token in text for token in FORBIDDEN_TEXT):
                fail(f"Local absolute path remains: {p.relative_to(ROOT)}")
        if p.suffix.lower() == ".json":
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                fail(f"Invalid JSON {p.relative_to(ROOT)}: {exc}")
        if p.suffix.lower() == ".py":
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except Exception as exc:
                fail(f"Python syntax error {p.relative_to(ROOT)}: {exc}")

    stages = list((ROOT / "lblock" / "stages").glob("[0-9][0-9]_*"))
    if len(stages) != 15:
        fail(f"Expected 15 canonical stages, found {len(stages)}")

    required = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "data" / "DATA_ARCHIVE_MANIFEST.csv",
        ROOT / "results" / "official_summaries" / "stage_15_shfa_summary.json",
    ]
    for p in required:
        if not p.is_file():
            fail(f"Missing required file: {p.relative_to(ROOT)}")

    print(f"[PASS] Repository validation succeeded: {len(files)} files, 15 stages.")

if __name__ == "__main__":
    main()
