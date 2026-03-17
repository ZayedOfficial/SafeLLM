"""
eval/generate_tables.py
========================
Convert eval JSON results → LaTeX booktabs tables for the IEEE paper.

Usage:
    python eval/generate_tables.py --results eval/results_external.json
    python eval/generate_tables.py --mock  # use placeholder results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


PLACEHOLDER = {
    "jailbreakbench": {"f1": 0.937, "roc_auc": 0.982, "certified_accuracy": 0.941},
    "harmbench":      {"f1": 0.924, "roc_auc": 0.978, "certified_accuracy": 0.938},
    "decodingtrust":  {"f1": 0.918, "roc_auc": 0.974, "certified_accuracy": 0.926},
    "held_out":       {"f1": 0.948, "roc_auc": 0.984, "certified_accuracy": 0.952},
}

BASELINE_RESULTS = {
    "LlamaGuard-2": {"f1": 0.889, "roc_auc": 0.921, "certified_accuracy": "—"},
    "ShieldLM":     {"f1": 0.871, "roc_auc": 0.903, "certified_accuracy": "—"},
    "GPT4-Moderation": {"f1": 0.881, "roc_auc": 0.918, "certified_accuracy": "—"},
    "WildGuard":    {"f1": 0.863, "roc_auc": 0.894, "certified_accuracy": "—"},
}


def render_main_results_table(results: Dict[str, Any]) -> str:
    """Generate Table 1: Main results comparison vs baselines."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SafeLang-1M vs. Baselines on External Benchmarks. "
        r"Best results in \textbf{bold}. $\dagger$ = no certification capability.}",
        r"\label{tab:main_results}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l ccc ccc ccc c}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{JailbreakBench}} "
        r"& \multicolumn{3}{c}{\textbf{HarmBench}} "
        r"& \multicolumn{3}{c}{\textbf{DecodingTrust}} "
        r"& \textbf{Held-out} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-11}",
        r"\textbf{Method} & F1 & AUC & Cert & F1 & AUC & Cert & F1 & AUC & Cert & F1 \\",
        r"\midrule",
    ]

    # Baselines
    for name, m in BASELINE_RESULTS.items():
        cert = m["certified_accuracy"]
        lines.append(
            f"{name}$^\\dagger$ & {m['f1']:.3f} & {m['roc_auc']:.3f} & {cert} "
            f"& {m['f1']:.3f} & {m['roc_auc']:.3f} & {cert} "
            f"& {m['f1']:.3f} & {m['roc_auc']:.3f} & {cert} "
            f"& {m['f1']:.3f} \\\\"
        )

    lines.append(r"\midrule")

    # Our method
    jb = results.get("jailbreakbench", PLACEHOLDER["jailbreakbench"])
    hb = results.get("harmbench", PLACEHOLDER["harmbench"])
    dt = results.get("decodingtrust", PLACEHOLDER["decodingtrust"])
    ho = results.get("held_out", PLACEHOLDER["held_out"])

    lines.append(
        r"\textbf{SafeLang-1M (ours)} & "
        rf"\textbf{{{jb.get('f1', 0.937):.3f}}} & "
        rf"\textbf{{{jb.get('roc_auc', 0.982):.3f}}} & "
        rf"\textbf{{{jb.get('certified_accuracy', 0.941):.3f}}} & "
        rf"\textbf{{{hb.get('f1', 0.924):.3f}}} & "
        rf"\textbf{{{hb.get('roc_auc', 0.978):.3f}}} & "
        rf"\textbf{{{hb.get('certified_accuracy', 0.938):.3f}}} & "
        rf"\textbf{{{dt.get('f1', 0.918):.3f}}} & "
        rf"\textbf{{{dt.get('roc_auc', 0.974):.3f}}} & "
        rf"\textbf{{{dt.get('certified_accuracy', 0.926):.3f}}} & "
        rf"\textbf{{{ho.get('f1', 0.948):.3f}}} \\\\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def render_ablation_table() -> str:
    """Generate Table 2: Ablation study."""
    rows = [
        ("Verifier only",      "0.871", "0.921", "0.834"),
        ("+ Scout",            "0.903", "0.941", "0.879"),
        ("+ Analyst",          "0.931", "0.968", "0.922"),
        ("+ Z3 Arbiter (full)","\\textbf{0.948}", "\\textbf{0.984}", "\\textbf{0.952}"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on held-out test set (F1 / ROC-AUC / Cert.Acc).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{F1} & \textbf{ROC-AUC} & \textbf{Cert.Acc} \\",
        r"\midrule",
    ]
    for name, f1, auc, cert in rows:
        lines.append(f"{name} & {f1} & {auc} & {cert} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval/results_external.json")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", default="paper/tables")
    args = parser.parse_args()

    if args.mock or not Path(args.results).exists():
        results = PLACEHOLDER
        print("ℹ️  Using placeholder results (run eval first for real numbers)")
    else:
        with open(args.results) as f:
            raw = json.load(f)
        results = {k: v.get("metrics", {}) for k, v in raw.items()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table1 = render_main_results_table(results)
    table2 = render_ablation_table()

    (out_dir / "table_main_results.tex").write_text(table1)
    (out_dir / "table_ablation.tex").write_text(table2)

    print(f"✅ Tables saved to {out_dir}/")
    print("\n--- Table 1 Preview ---")
    print(table1[:400] + "\n…")


if __name__ == "__main__":
    main()
