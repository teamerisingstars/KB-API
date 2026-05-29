"""Validator script used by action.yml.

Builds the BM25 index against the caller's docs path, runs each question from
the supplied questions file, prints a markdown summary, sets action outputs,
and exits non-zero if --fail-on-missing is true and any question missed.
"""
import json
import os
import sys
from pathlib import Path

# The action's Dockerfile WORKDIR is /app, where app/ lives.
sys.path.insert(0, "/app")

from app.indexer import IndexStore, build_index
from app.searcher import search


def _set_output(name: str, value: str) -> None:
    """Write a step output to GITHUB_OUTPUT (multiline-safe)."""
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    delim = "EOF_KBAPI_OUT"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"{name}<<{delim}\n{value}\n{delim}\n")


def _append_summary(md: str) -> None:
    """Write to GITHUB_STEP_SUMMARY so the result shows up in the run UI."""
    out_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(md + "\n")


def _parse_line(line: str) -> tuple[str, str | None]:
    """Each line is either a plain question or `{"q":"...","must_match":"..."}`."""
    line = line.strip()
    if not line or line.startswith("#"):
        return "", None
    if line.startswith("{"):
        try:
            obj = json.loads(line)
            return obj.get("q", "").strip(), obj.get("must_match")
        except json.JSONDecodeError:
            return line, None
    return line, None


def main() -> int:
    docs_path = sys.argv[1] if len(sys.argv) > 1 else "docs/"
    questions_file = sys.argv[2] if len(sys.argv) > 2 else ".github/docs-questions.txt"
    fail_on_missing = (sys.argv[3] if len(sys.argv) > 3 else "true").lower() == "true"
    min_conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

    workspace = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    docs_dir = (workspace / docs_path).resolve()
    q_path = (workspace / questions_file).resolve()

    if not docs_dir.exists():
        print(f"::error::Docs path not found: {docs_dir}")
        return 2
    if not q_path.exists():
        print(f"::error::Questions file not found: {q_path}")
        return 2

    questions: list[tuple[str, str | None]] = []
    for raw in q_path.read_text(encoding="utf-8").splitlines():
        q, must = _parse_line(raw)
        if q:
            questions.append((q, must))

    if not questions:
        print("::warning::No questions found in file; nothing to validate.")
        _set_output("results-json", "[]")
        _set_output("pass-count", "0")
        _set_output("fail-count", "0")
        _set_output("summary", "0 of 0 passed")
        return 0

    print(f"Indexing {docs_dir} …")
    store = IndexStore()
    build_index(str(docs_dir), store)
    print(f"  indexed {len(store.sections)} sections from {store.file_count} files")

    results = []
    passes = 0
    fails = 0
    for q, must_match in questions:
        resp = search(q, store)
        answer = resp.answer
        section = resp.section
        source = resp.source or ""
        conf = resp.confidence or 0.0

        passed = answer is not None and conf >= min_conf
        if passed and must_match:
            passed = must_match.lower() in source.lower() or must_match.lower() in (section or "").lower()

        results.append({
            "question": q,
            "section": section,
            "source": source,
            "confidence": round(conf, 4),
            "passed": passed,
        })
        if passed:
            passes += 1
        else:
            fails += 1

    summary_line = f"{passes} of {len(questions)} passed"
    print(f"\n{summary_line}")

    # Markdown summary visible in the GitHub Actions run UI
    md = ["## Docs Q&A validation", "", f"**{summary_line}**  (min confidence: {min_conf})", "",
          "| # | Question | Section | Source | Confidence | Result |",
          "|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        sect = (r["section"] or "—")[:40]
        src = (r["source"] or "—")[:40]
        mark = "✅" if r["passed"] else "❌"
        md.append(f"| {i} | {r['question'][:50]} | {sect} | `{src}` | {r['confidence']} | {mark} |")
    _append_summary("\n".join(md))

    _set_output("results-json", json.dumps(results))
    _set_output("pass-count", str(passes))
    _set_output("fail-count", str(fails))
    _set_output("summary", summary_line)

    if fail_on_missing and fails > 0:
        print(f"::error::{fails} question(s) failed validation")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
