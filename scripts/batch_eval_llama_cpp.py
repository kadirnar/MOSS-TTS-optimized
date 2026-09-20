"""
Batch inference for TTS evaluation benchmarks (Seed-TTS-eval & CV3-eval).

Uses the llama.cpp backend (MOSS-TTS-Delay).

Expected benchmark layout (per case)::

    {benchmark_dir}/{task}/{case_id}/prompt.wav
    {benchmark_dir}/{task}/{case_id}/label.txt

Output layout::

    {result_dir}/{task}/{case_id}/pred.wav

Usage::

    python scripts/batch_eval_llama_cpp.py \\
        --config configs/llama_cpp/default.yaml \\
        --benchmark-dir /path/to/eval/tts \\
        --result-dir results/my_run \\
        --tasks seed-tts-zeroshot-zh seed-tts-zeroshot-en
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig
from moss_tts_delay.llama_cpp._constants import SAMPLE_RATE

log = logging.getLogger(__name__)

SEED_TTS_TASKS = [
    "seed-tts-zeroshot-zh",
    "seed-tts-zeroshot-en",
    "seed-tts-zeroshot-hard-zh",
]

CV3_TASKS = [
    "cv3-crosslingual-en",
    "cv3-crosslingual-hard-en",
    "cv3-zeroshot-en",
    "cv3-zeroshot-hard-en",
    "cv3-crosslingual-zh",
    "cv3-crosslingual-hard-zh",
    "cv3-zeroshot-zh",
    "cv3-zeroshot-hard-zh",
]

ALL_TASKS = SEED_TTS_TASKS + CV3_TASKS + ["demo-zh", "demo-en"]

TASK_LANGUAGE = {
    "seed-tts-zeroshot-zh": "zh",
    "seed-tts-zeroshot-en": "en",
    "seed-tts-zeroshot-hard-zh": "zh",
    "cv3-crosslingual-en": "en",
    "cv3-crosslingual-hard-en": "en",
    "cv3-zeroshot-en": "en",
    "cv3-zeroshot-hard-en": "en",
    "cv3-crosslingual-zh": "zh",
    "cv3-crosslingual-hard-zh": "zh",
    "cv3-zeroshot-zh": "zh",
    "cv3-zeroshot-hard-zh": "zh",
    "demo-zh": "zh",
    "demo-en": "en",
}


@dataclass
class CaseResult:
    task: str
    case_id: str
    success: bool
    audio_duration: float = 0.0
    generation_time: float = 0.0
    error: str = ""


def discover_cases(benchmark_dir: Path, tasks: list[str]) -> list[tuple[str, str, Path, str]]:
    cases = []
    for task in tasks:
        task_dir = benchmark_dir / task
        if not task_dir.is_dir():
            log.warning("Task directory not found: %s", task_dir)
            continue
        for case_dir in sorted(task_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            prompt_wav = case_dir / "prompt.wav"
            label_txt = case_dir / "label.txt"
            if not label_txt.exists():
                log.warning("Missing label.txt: %s", case_dir)
                continue
            text = label_txt.read_text().strip()
            cases.append((task, case_dir.name, prompt_wav, text))
    return cases


def run_batch(
    pipeline: LlamaCppPipeline,
    cases: list[tuple[str, str, Path, str]],
    result_dir: Path,
    max_cases: int = 0,
    skip_existing: bool = True,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    total = len(cases) if max_cases <= 0 else min(max_cases, len(cases))
    cases = cases[:total]

    log.info("Running %d evaluation cases, output -> %s", total, result_dir)

    pbar = tqdm(cases, desc="Evaluation", unit="case", total=total, dynamic_ncols=True)
    for i, (task, case_id, prompt_wav, text) in enumerate(pbar):
        pbar.set_postfix_str(f"{task}/{case_id}")
        out_dir = result_dir / task / case_id
        out_wav = out_dir / "pred.wav"

        if skip_existing and out_wav.exists():
            log.info("[%d/%d] %s/%s — skipped (exists)", i + 1, total, task, case_id)
            results.append(CaseResult(task=task, case_id=case_id, success=True))
            continue

        log.info("[%d/%d] %s/%s — %s", i + 1, total, task, case_id, text[:60])
        t0 = time.time()

        try:
            lang = TASK_LANGUAGE.get(task)
            ref_audio = str(prompt_wav) if prompt_wav.exists() else None

            waveform = pipeline.generate(
                text=text, reference_audio=ref_audio, language=lang,
            )
            elapsed = time.time() - t0

            if waveform.size == 0:
                results.append(CaseResult(
                    task=task, case_id=case_id, success=False,
                    generation_time=elapsed, error="empty waveform",
                ))
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_wav), waveform, SAMPLE_RATE)
            audio_dur = len(waveform) / SAMPLE_RATE

            results.append(CaseResult(
                task=task, case_id=case_id, success=True,
                audio_duration=audio_dur, generation_time=elapsed,
            ))
            log.info(
                "  -> %.2fs audio in %.2fs (RTF=%.2f)",
                audio_dur, elapsed, elapsed / max(audio_dur, 1e-6),
            )

        except Exception as e:
            elapsed = time.time() - t0
            log.error("  -> FAILED: %s", e)
            results.append(CaseResult(
                task=task, case_id=case_id, success=False,
                generation_time=elapsed, error=str(e),
            ))

    return results


def write_summary(results: list[CaseResult], result_dir: Path) -> None:
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    per_task: dict[str, dict] = {}
    for r in results:
        if r.task not in per_task:
            per_task[r.task] = {"total": 0, "success": 0, "failed": 0, "total_audio_s": 0.0, "total_gen_s": 0.0}
        per_task[r.task]["total"] += 1
        if r.success:
            per_task[r.task]["success"] += 1
            per_task[r.task]["total_audio_s"] += r.audio_duration
            per_task[r.task]["total_gen_s"] += r.generation_time
        else:
            per_task[r.task]["failed"] += 1

    for task, stats in per_task.items():
        if stats["total_audio_s"] > 0:
            stats["avg_rtf"] = round(stats["total_gen_s"] / stats["total_audio_s"], 3)

    summary = {
        "total_cases": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "per_task": per_task,
    }

    if failed:
        summary["failures"] = [
            {"task": r.task, "case_id": r.case_id, "error": r.error}
            for r in failed
        ]

    summary_path = result_dir / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Summary written to %s", summary_path)

    print("\n" + "=" * 60)
    print("  BATCH INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Total:     {len(results)}")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    for task, stats in per_task.items():
        rtf = stats.get("avg_rtf", "N/A")
        print(f"  {task}: {stats['success']}/{stats['total']}  RTF={rtf}")
    print("=" * 60 + "\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Batch TTS evaluation (llama.cpp backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML config")
    parser.add_argument(
        "--benchmark-dir",
        default="/inspire/hdd/project/embodied-multimodality/public/speech_generation/data/eval/tts",
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--suite", choices=["seed-tts", "cv3", "all"], default=None)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--no-skip", action="store_true")

    parser.add_argument("--text-temp", type=float, default=None)
    parser.add_argument("--audio-temp", type=float, default=None)
    parser.add_argument("--audio-top-p", type=float, default=None)
    parser.add_argument("--audio-top-k", type=int, default=None)
    parser.add_argument("--audio-rep-penalty", type=float, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--heads-backend", choices=["auto", "numpy", "torch"], default=None)

    args = parser.parse_args()
    config = PipelineConfig.from_yaml(args.config)

    if args.text_temp is not None:
        config.text_temperature = args.text_temp
    if args.audio_temp is not None:
        config.audio_temperature = args.audio_temp
    if args.audio_top_p is not None:
        config.audio_top_p = args.audio_top_p
    if args.audio_top_k is not None:
        config.audio_top_k = args.audio_top_k
    if args.audio_rep_penalty is not None:
        config.audio_repetition_penalty = args.audio_rep_penalty
    if args.n_gpu_layers is not None:
        config.n_gpu_layers = args.n_gpu_layers
    if args.max_tokens is not None:
        config.max_new_tokens = args.max_tokens
    if args.heads_backend is not None:
        config.heads_backend = args.heads_backend

    if args.tasks:
        tasks = args.tasks
    elif args.suite == "seed-tts":
        tasks = SEED_TTS_TASKS
    elif args.suite == "cv3":
        tasks = CV3_TASKS
    else:
        tasks = ALL_TASKS

    for t in tasks:
        if t not in ALL_TASKS:
            log.error("Unknown task: %s. Valid tasks: %s", t, ALL_TASKS)
            sys.exit(1)

    benchmark_dir = Path(args.benchmark_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(benchmark_dir, tasks)
    if not cases:
        log.error("No cases found in %s for tasks %s", benchmark_dir, tasks)
        sys.exit(1)
    log.info("Discovered %d cases across %d tasks", len(cases), len(tasks))

    run_meta = {
        "config": args.config,
        "benchmark_dir": str(benchmark_dir),
        "tasks": tasks,
        "sampling": {
            "text_temperature": config.text_temperature,
            "text_top_p": config.text_top_p,
            "text_top_k": config.text_top_k,
            "audio_temperature": config.audio_temperature,
            "audio_top_p": config.audio_top_p,
            "audio_top_k": config.audio_top_k,
            "audio_repetition_penalty": config.audio_repetition_penalty,
        },
        "max_new_tokens": config.max_new_tokens,
        "backbone_gguf": config.backbone_gguf,
        "heads_backend": config.heads_backend,
    }
    with open(result_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    with LlamaCppPipeline(config) as pipeline:
        results = run_batch(
            pipeline, cases, result_dir,
            max_cases=args.max_cases,
            skip_existing=not args.no_skip,
        )

    write_summary(results, result_dir)


if __name__ == "__main__":
    main()
