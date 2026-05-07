from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


# ============================================================
# Helpers
# ============================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_ascii_token(text: Any) -> str:
    if text is None:
        return ""
    out = "".join(ch.lower() for ch in str(text).strip() if ch.isalnum())
    return out


def load_csv_rows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = [tok for tok in str(reference).strip().split() if tok]
    hyp = [tok for tok in str(hypothesis).strip().split() if tok]
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, ca in enumerate(ref, start=1):
        curr = [i]
        for j, cb in enumerate(hyp, start=1):
            curr.append(min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return float(prev[-1]) / float(len(ref))


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = str(reference)
    hyp = str(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(levenshtein(ref, hyp)) / float(len(ref))


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute a simple smoothed BLEU score for one transcript pair."""
    ref_tokens = [tok for tok in str(reference).lower().split() if tok]
    hyp_tokens = [tok for tok in str(hypothesis).lower().split() if tok]

    if not ref_tokens:
        return 1.0 if not hyp_tokens else 0.0
    if not hyp_tokens:
        return 0.0

    max_order = max(1, min(max_n, len(ref_tokens), len(hyp_tokens)))
    precisions: list[float] = []

    for n in range(1, max_order + 1):
        ref_counts: dict[tuple[str, ...], int] = {}
        hyp_counts: dict[tuple[str, ...], int] = {}

        for gram in ngrams(ref_tokens, n):
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
        for gram in ngrams(hyp_tokens, n):
            hyp_counts[gram] = hyp_counts.get(gram, 0) + 1

        overlap = 0
        total = 0
        for gram, count in hyp_counts.items():
            overlap += min(count, ref_counts.get(gram, 0))
            total += count

        precisions.append((overlap + 1.0) / (total + 1.0))

    if len(hyp_tokens) > len(ref_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (len(ref_tokens) / len(hyp_tokens)))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return float(brevity_penalty * geo_mean)


# ============================================================
# Demo integration
# ============================================================


@dataclass
class BackendRun:
    raw_input: str
    expected_output: str
    mode: str
    transcript_snapshot: str
    fuzzy_output: str
    gemma_output: str
    openai_output: str
    final_output: str
    correction_source: str
    changed: bool
    fuzzy_time_ms: float
    gemma_time_ms: float
    openai_time_ms: float
    total_time_ms: float
    openai_available: bool
    openai_attempted: bool
    openai_status: str


class DemoBridge:
    def __init__(self, prefer_openai: bool = False, disable_local_llm: bool = False, disable_openai: bool = False):
        self.module = self._load_module()
        cfg = self.module.DemoConfig()
        cfg.ai_backend = "openai" if prefer_openai else cfg.ai_backend
        cfg.local_llm_enabled = not bool(disable_local_llm)
        cfg.openai_fallback_enabled = not bool(disable_openai)
        self.corrector = self.module.HybridAutoCorrector(cfg)

    @staticmethod
    def _load_module():
        import sys

        current_dir = Path(__file__).resolve().parent
        candidates = [
            current_dir / "Sign-Bridge-Demo.py",
            current_dir / "Sign-Bridge-Demo(32).py",
        ]
        script_path = next((path for path in candidates if path.exists()), None)
        if script_path is None:
            names = ", ".join(path.name for path in candidates)
            raise FileNotFoundError(f"Missing demo file. Expected one of: {names}")

        module_name = "sign_bridge_demo"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {script_path.name}")

        module = importlib.util.module_from_spec(spec)

        # Register the dynamically loaded module before executing it.
        # This is required for @dataclass classes inside Sign-Bridge-Demo.py.
        sys.modules[module_name] = module

        spec.loader.exec_module(module)
        return module

    def run_autocorrect(self, raw_input: str, expected_output: str = "", mode: str = "letter", transcript_snapshot: str = "") -> BackendRun:
        raw = self.module.sanitize_token_text(raw_input)
        local = self.module.normalize_token_for_match(self.corrector.local_correct(raw))
        fuzzy_time_ms = 0.0
        gemma_output = ""
        gemma_time_ms = 0.0
        openai_output = ""
        openai_time_ms = 0.0
        openai_attempted = False
        openai_status = self.corrector.openai_client.status
        source = "Local"
        final = raw
        changed = False

        # Measure fuzzy alone using the same logic as the demo.
        start = self.module.time.time()
        local = self.module.normalize_token_for_match(self.corrector.local_correct(raw))
        fuzzy_time_ms = (self.module.time.time() - start) * 1000.0

        if local and local != raw:
            final = local
            source = "Fuzzy"
            changed = True
        elif self.corrector.should_try_ai(raw, local):
            start_ai = self.module.time.time()
            ai_text, backend_label, trace = self.corrector._ai_correct_sync(raw)
            _ = (self.module.time.time() - start_ai) * 1000.0
            gemma_output = self.module.sanitize_token_text(str(trace.get("gemma_output", "") or ""))
            gemma_time_ms = float(trace.get("gemma_time_ms", 0.0) or 0.0)
            openai_output = self.module.sanitize_token_text(str(trace.get("openai_output", "") or ""))
            openai_time_ms = float(trace.get("openai_time_ms", 0.0) or 0.0)
            openai_attempted = bool(trace.get("openai_attempted", False))
            openai_status = str(trace.get("openai_status", self.corrector.openai_client.status) or self.corrector.openai_client.status)
            ai_text = self.module.sanitize_token_text(ai_text, fallback=raw)
            if ai_text and ai_text != raw:
                final = ai_text
                source = "Gemma" if backend_label == "Local Gemma" else ("OpenAI" if backend_label == "OpenAI" else "Local")
                changed = True

        total_time_ms = float(fuzzy_time_ms + gemma_time_ms + openai_time_ms)
        return BackendRun(
            raw_input=raw,
            expected_output=self.module.sanitize_token_text(expected_output),
            mode=mode,
            transcript_snapshot=self.module.sanitize_text_for_display(transcript_snapshot, max_len=180),
            fuzzy_output=local or raw,
            gemma_output=gemma_output,
            openai_output=openai_output,
            final_output=final,
            correction_source=source,
            changed=changed,
            fuzzy_time_ms=fuzzy_time_ms,
            gemma_time_ms=gemma_time_ms,
            openai_time_ms=openai_time_ms,
            total_time_ms=total_time_ms,
            openai_available=bool(self.corrector.openai_client.available),
            openai_attempted=openai_attempted,
            openai_status=self.module.sanitize_text_for_display(openai_status, max_len=180),
        )


# ============================================================
# Evaluation logic
# ============================================================


@dataclass
class Aggregate:
    total: int = 0
    raw_correct: int = 0
    final_correct: int = 0
    fuzzy_correct: int = 0
    gemma_correct: int = 0
    openai_correct: int = 0
    keep_correct: int = 0
    fix_correct: int = 0
    harm_count: int = 0
    miss_count: int = 0
    changed_count: int = 0
    avg_total_ms: float = 0.0
    avg_fuzzy_ms: float = 0.0
    avg_gemma_ms: float = 0.0
    avg_openai_ms: float = 0.0
    avg_final_cer: float = 0.0
    avg_final_wer: float = 0.0
    avg_final_bleu: float = 0.0


def evaluate_runs(runs: list[BackendRun]) -> Aggregate:
    agg = Aggregate(total=len(runs))
    if not runs:
        return agg

    total_ms = []
    fuzzy_ms = []
    gemma_ms = []
    openai_ms = []
    final_cers = []
    final_wers = []
    final_bleus = []

    for run in runs:
        raw_ok = normalize_ascii_token(run.raw_input) == normalize_ascii_token(run.expected_output)
        fuzzy_ok = normalize_ascii_token(run.fuzzy_output) == normalize_ascii_token(run.expected_output)
        gemma_ok = normalize_ascii_token(run.gemma_output) == normalize_ascii_token(run.expected_output) if run.gemma_output else False
        openai_ok = normalize_ascii_token(run.openai_output) == normalize_ascii_token(run.expected_output) if run.openai_output else False
        final_ok = normalize_ascii_token(run.final_output) == normalize_ascii_token(run.expected_output)

        agg.raw_correct += int(raw_ok)
        agg.fuzzy_correct += int(fuzzy_ok)
        agg.gemma_correct += int(gemma_ok)
        agg.openai_correct += int(openai_ok)
        agg.final_correct += int(final_ok)
        agg.changed_count += int(run.changed)

        if raw_ok and final_ok:
            agg.keep_correct += 1
        elif (not raw_ok) and final_ok:
            agg.fix_correct += 1
        elif raw_ok and (not final_ok):
            agg.harm_count += 1
        elif (not raw_ok) and (not final_ok):
            agg.miss_count += 1

        total_ms.append(run.total_time_ms)
        fuzzy_ms.append(run.fuzzy_time_ms)
        gemma_ms.append(run.gemma_time_ms)
        openai_ms.append(run.openai_time_ms)
        final_cers.append(char_error_rate(run.expected_output, run.final_output))
        final_wers.append(word_error_rate(run.expected_output, run.final_output))
        final_bleus.append(bleu_score(run.expected_output, run.final_output))

    agg.avg_total_ms = statistics.fmean(total_ms) if total_ms else 0.0
    agg.avg_fuzzy_ms = statistics.fmean(fuzzy_ms) if fuzzy_ms else 0.0
    agg.avg_gemma_ms = statistics.fmean(gemma_ms) if gemma_ms else 0.0
    agg.avg_openai_ms = statistics.fmean(openai_ms) if openai_ms else 0.0
    agg.avg_final_cer = statistics.fmean(final_cers) if final_cers else 0.0
    agg.avg_final_wer = statistics.fmean(final_wers) if final_wers else 0.0
    agg.avg_final_bleu = statistics.fmean(final_bleus) if final_bleus else 0.0
    return agg


# ============================================================
# Commands
# ============================================================


def cmd_make_templates(args: argparse.Namespace) -> None:
    correction_rows = [
        {
            "sample_id": "1",
            "mode": "letter",
            "raw_input": "fac",
            "expected_output": "face",
            "transcript_snapshot": "",
            "notes": "simple fuzzy-style fix",
        },
        {
            "sample_id": "2",
            "mode": "letter",
            "raw_input": "loow",
            "expected_output": "slow",
            "transcript_snapshot": "",
            "notes": "missing leading character example",
        },
        {
            "sample_id": "3",
            "mode": "letter",
            "raw_input": "tall",
            "expected_output": "tall",
            "transcript_snapshot": "",
            "notes": "already correct; should not be harmed",
        },
    ]
    transcript_rows = [
        {
            "sample_id": "1",
            "mode": "letter",
            "target_transcript": "hello world",
            "observed_transcript": "hello world",
            "notes": "perfect transcript example",
        },
        {
            "sample_id": "2",
            "mode": "letter",
            "target_transcript": "thank you",
            "observed_transcript": "thnak you",
            "notes": "use for CER/WER tracking",
        },
    ]
    write_csv(
        args.correction_out,
        ["sample_id", "mode", "raw_input", "expected_output", "transcript_snapshot", "notes"],
        correction_rows,
    )
    write_csv(
        args.transcript_out,
        ["sample_id", "mode", "target_transcript", "observed_transcript", "notes"],
        transcript_rows,
    )
    print(f"Wrote correction template -> {args.correction_out}")
    print(f"Wrote transcript template -> {args.transcript_out}")



def cmd_eval_corrections(args: argparse.Namespace) -> None:
    rows = load_csv_rows(args.benchmark)
    bridge = DemoBridge(
        prefer_openai=bool(args.prefer_openai),
        disable_local_llm=bool(args.disable_local_llm),
        disable_openai=bool(args.disable_openai),
    )

    runs: list[BackendRun] = []
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        raw_input = str(row.get("raw_input", "") or "").strip()
        expected_output = str(row.get("expected_output", "") or "").strip()
        if not raw_input or not expected_output:
            continue
        mode = str(row.get("mode", "letter") or "letter").strip() or "letter"
        transcript_snapshot = str(row.get("transcript_snapshot", "") or "")
        run = bridge.run_autocorrect(
            raw_input=raw_input,
            expected_output=expected_output,
            mode=mode,
            transcript_snapshot=transcript_snapshot,
        )
        runs.append(run)
        raw_ok = normalize_ascii_token(run.raw_input) == normalize_ascii_token(run.expected_output)
        final_ok = normalize_ascii_token(run.final_output) == normalize_ascii_token(run.expected_output)
        out_rows.append({
            "sample_id": row.get("sample_id", str(idx)),
            "mode": run.mode,
            "raw_input": run.raw_input,
            "expected_output": run.expected_output,
            "fuzzy_output": run.fuzzy_output,
            "gemma_output": run.gemma_output,
            "openai_output": run.openai_output,
            "final_output": run.final_output,
            "correction_source": run.correction_source,
            "raw_correct": int(raw_ok),
            "final_correct": int(final_ok),
            "changed": int(run.changed),
            "fuzzy_time_ms": f"{run.fuzzy_time_ms:.3f}",
            "gemma_time_ms": f"{run.gemma_time_ms:.3f}",
            "openai_time_ms": f"{run.openai_time_ms:.3f}",
            "total_time_ms": f"{run.total_time_ms:.3f}",
            "openai_available": int(run.openai_available),
            "openai_attempted": int(run.openai_attempted),
            "openai_status": run.openai_status,
        })

    if args.results_out:
        write_csv(
            args.results_out,
            [
                "sample_id", "mode", "raw_input", "expected_output", "fuzzy_output", "gemma_output",
                "openai_output", "final_output", "correction_source", "raw_correct", "final_correct",
                "changed", "fuzzy_time_ms", "gemma_time_ms", "openai_time_ms", "total_time_ms",
                "openai_available", "openai_attempted", "openai_status",
            ],
            out_rows,
        )

    agg = evaluate_runs(runs)
    print(f"Samples: {agg.total}")
    if agg.total <= 0:
        return
    print(f"Raw accuracy:   {agg.raw_correct / agg.total:.3f} ({agg.raw_correct}/{agg.total})")
    print(f"Fuzzy accuracy: {agg.fuzzy_correct / agg.total:.3f} ({agg.fuzzy_correct}/{agg.total})")
    print(f"Gemma accuracy: {agg.gemma_correct / agg.total:.3f} ({agg.gemma_correct}/{agg.total})")
    print(f"OpenAI accuracy:{agg.openai_correct / agg.total:.3f} ({agg.openai_correct}/{agg.total})")
    print(f"Final accuracy: {agg.final_correct / agg.total:.3f} ({agg.final_correct}/{agg.total})")
    print(f"Keep correct:   {agg.keep_correct}")
    print(f"Fix correct:    {agg.fix_correct}")
    print(f"Harm count:     {agg.harm_count}")
    print(f"Miss count:     {agg.miss_count}")
    print(f"Changed count:  {agg.changed_count}")
    print(f"Avg total ms:   {agg.avg_total_ms:.3f}")
    print(f"Avg fuzzy ms:   {agg.avg_fuzzy_ms:.3f}")
    print(f"Avg gemma ms:   {agg.avg_gemma_ms:.3f}")
    print(f"Avg openai ms:  {agg.avg_openai_ms:.3f}")
    print(f"Avg final CER:  {agg.avg_final_cer:.3f}")
    print(f"Avg final WER:  {agg.avg_final_wer:.3f}")
    print(f"Avg final BLEU: {agg.avg_final_bleu:.3f}")
    if args.results_out:
        print(f"Detailed results -> {args.results_out}")



def cmd_score_logs(args: argparse.Namespace) -> None:
    logs = load_csv_rows(args.log_csv)
    gold_rows = load_csv_rows(args.ground_truth_csv)
    gold_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in gold_rows:
        raw = normalize_ascii_token(row.get("raw_input", ""))
        mode = str(row.get("mode", "letter") or "letter").strip() or "letter"
        if raw:
            gold_map[(mode, raw)] = row

    matched_runs: list[BackendRun] = []
    unmatched = 0
    for row in logs:
        raw = normalize_ascii_token(row.get("raw_input", ""))
        mode = str(row.get("mode", "letter") or "letter").strip() or "letter"
        gold = gold_map.get((mode, raw))
        if gold is None:
            unmatched += 1
            continue
        matched_runs.append(BackendRun(
            raw_input=str(row.get("raw_input", "") or ""),
            expected_output=str(gold.get("expected_output", "") or ""),
            mode=mode,
            transcript_snapshot=str(row.get("transcript_snapshot", "") or ""),
            fuzzy_output=str(row.get("fuzzy_output", "") or ""),
            gemma_output=str(row.get("gemma_output", "") or ""),
            openai_output=str(row.get("openai_output", "") or ""),
            final_output=str(row.get("final_output", "") or ""),
            correction_source=str(row.get("correction_source", "Local") or "Local"),
            changed=bool(int(safe_float(row.get("changed", 0), 0.0))),
            fuzzy_time_ms=safe_float(row.get("fuzzy_time_ms", 0.0), 0.0),
            gemma_time_ms=safe_float(row.get("gemma_time_ms", 0.0), 0.0),
            openai_time_ms=safe_float(row.get("openai_time_ms", 0.0), 0.0),
            total_time_ms=safe_float(row.get("total_time_ms", 0.0), 0.0),
            openai_available=bool(int(safe_float(row.get("openai_available", 0), 0.0))),
            openai_attempted=bool(int(safe_float(row.get("openai_attempted", 0), 0.0))),
            openai_status=str(row.get("openai_status", "") or ""),
        ))

    agg = evaluate_runs(matched_runs)
    print(f"Matched log rows: {agg.total}")
    print(f"Unmatched log rows: {unmatched}")
    if agg.total <= 0:
        return
    print(f"Final accuracy: {agg.final_correct / agg.total:.3f} ({agg.final_correct}/{agg.total})")
    print(f"Keep correct:   {agg.keep_correct}")
    print(f"Fix correct:    {agg.fix_correct}")
    print(f"Harm count:     {agg.harm_count}")
    print(f"Miss count:     {agg.miss_count}")
    print(f"Avg total ms:   {agg.avg_total_ms:.3f}")
    print(f"Avg final CER:  {agg.avg_final_cer:.3f}")
    print(f"Avg final WER:  {agg.avg_final_wer:.3f}")
    print(f"Avg final BLEU: {agg.avg_final_bleu:.3f}")


def summarize_numbers(values: list[float]) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0.0, "avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(clean)),
        "avg": statistics.fmean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def print_latency_line(label: str, values: list[float]) -> None:
    stats = summarize_numbers(values)
    print(
        f"{label}: "
        f"avg={stats['avg']:.3f} ms, "
        f"median={stats['median']:.3f} ms, "
        f"min={stats['min']:.3f} ms, "
        f"max={stats['max']:.3f} ms"
    )


def cmd_summarize_latency(args: argparse.Namespace) -> None:
    rows = load_csv_rows(args.log_csv)
    if not rows:
        print("No latency rows found.")
        return

    total_ms = [safe_float(row.get("total_time_ms", 0.0), 0.0) for row in rows]
    fuzzy_ms = [safe_float(row.get("fuzzy_time_ms", 0.0), 0.0) for row in rows]
    gemma_ms = [safe_float(row.get("gemma_time_ms", 0.0), 0.0) for row in rows]
    openai_ms = [safe_float(row.get("openai_time_ms", 0.0), 0.0) for row in rows]

    print(f"Latency samples: {len(rows)}")
    print_latency_line("Total latency", total_ms)
    print_latency_line("Fuzzy latency", fuzzy_ms)
    print_latency_line("Gemma latency", gemma_ms)
    print_latency_line("OpenAI latency", openai_ms)

    by_source: dict[str, list[float]] = {}
    for row in rows:
        source = str(row.get("correction_source", "Unknown") or "Unknown")
        by_source.setdefault(source, []).append(safe_float(row.get("total_time_ms", 0.0), 0.0))

    if by_source:
        print("Latency by correction source:")
        for source in sorted(by_source):
            print_latency_line(f"  {source}", by_source[source])


def cmd_eval_transcripts(args: argparse.Namespace) -> None:
    rows = load_csv_rows(args.benchmark)
    total = 0
    exact = 0
    cers: list[float] = []
    wers: list[float] = []
    bleus: list[float] = []

    for row in rows:
        target = str(row.get("target_transcript", "") or "")
        observed = str(row.get("observed_transcript", "") or "")
        if not target:
            continue

        total += 1
        exact += int(target.strip().lower() == observed.strip().lower())
        cers.append(char_error_rate(target, observed))
        wers.append(word_error_rate(target, observed))
        bleus.append(bleu_score(target, observed))

    print(f"Transcript samples: {total}")
    if total <= 0:
        return

    print(f"Exact transcript accuracy: {exact / total:.3f} ({exact}/{total})")
    print(f"Average CER: {statistics.fmean(cers):.3f}")
    print(f"Average WER: {statistics.fmean(wers):.3f}")
    print(f"Average BLEU: {statistics.fmean(bleus):.3f}")


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate Sign-Bridge correction and transcript quality.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_templates = sub.add_parser("make-templates", help="Write starter CSV templates for correction and transcript evaluation.")
    p_templates.add_argument("--correction-out", default=os.path.join("data", "correction_benchmark_template.csv"))
    p_templates.add_argument("--transcript-out", default=os.path.join("data", "transcript_benchmark_template.csv"))
    p_templates.set_defaults(func=cmd_make_templates)

    p_corr = sub.add_parser("eval-corrections", help="Run the current hybrid autocorrector on a benchmark CSV with ground truth.")
    p_corr.add_argument("--benchmark", required=True, help="CSV with raw_input and expected_output columns.")
    p_corr.add_argument("--results-out", default=os.path.join("logs", "correction_eval_results.csv"))
    p_corr.add_argument("--prefer-openai", action="store_true", help="Try OpenAI before local Gemma when both are available.")
    p_corr.add_argument("--disable-local-llm", action="store_true", help="Disable the local Gemma backend for this evaluation run.")
    p_corr.add_argument("--disable-openai", action="store_true", help="Disable OpenAI fallback for this evaluation run.")
    p_corr.set_defaults(func=cmd_eval_corrections)

    p_logs = sub.add_parser("score-logs", help="Score an existing sign_bridge_performance.csv file against a ground-truth benchmark CSV.")
    p_logs.add_argument("--log-csv", required=True, help="Existing performance log CSV from the demo.")
    p_logs.add_argument("--ground-truth-csv", required=True, help="Correction benchmark CSV containing expected_output.")
    p_logs.set_defaults(func=cmd_score_logs)

    p_trans = sub.add_parser("eval-transcripts", help="Compute exact match, CER, WER, and BLEU for end-to-end transcript outputs.")
    p_trans.add_argument("--benchmark", required=True, help="CSV with target_transcript and observed_transcript columns.")
    p_trans.set_defaults(func=cmd_eval_transcripts)

    p_latency = sub.add_parser("summarize-latency", help="Summarize latency from logs/sign_bridge_performance.csv.")
    p_latency.add_argument("--log-csv", default=os.path.join("logs", "sign_bridge_performance.csv"))
    p_latency.set_defaults(func=cmd_summarize_latency)

    return ap


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
