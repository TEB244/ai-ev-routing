#!/usr/bin/env python3
"""
compute_wilcoxon.py -- resolve the Table 3 (tab:resource_analysis) \\pending{} note.

Computes the paired Wilcoxon signed-rank tests over the per-experiment final
rewards in the v2 cache (paper_figure_generators_v2/table_data/_v2_cache/
reward_per_exp.csv, built by build_v2_cache.py), prints paste-ready LaTeX, and
can patch the \\pending{} placeholders in the VERDE repo directly.

Two test designs are computed:

  1. PER-SEED (the paper's stated design, "the nine per-seed final rewards"):
     each DM is collapsed to ONE final reward per seed (mean over that seed's
     seasons / aggregation levels), then DM pairs are compared with a paired
     two-sided Wilcoxon signed-rank test over the shared seeds.
     NOTE: n seeds bounds the smallest achievable two-sided p at 2^-(n-1),
     so 3 seeds (4xxx only) gives min p = 0.25 -- underpowered. The full
     4xxx+5xxx+6xxx batch (9 seeds) gives min p ~= 0.004.

  2. PER-CONFIGURATION (secondary / robustness): DM pairs are matched on the
     (season, aggregation level, seed) configuration and the paired test runs
     over those matched configs (36 pairs with 3 seeds, 108 with 9). This is
     better powered and usable NOW with 4xxx alone; report it alongside (1)
     or use its wording if the 5xxx/6xxx batches do not land before the
     revision deadline.

Holm-Bonferroni adjusted p-values across the six DM pairs are reported for
both designs.

Usage (on huron, after building/refreshing the cache):
    # 3-seed preview (4xxx only):
    python paper_figure_generators_v2/build_v2_cache.py --metrics-root /storage_1/metrics_postfix
    # full 9 seeds once 5xxx/6xxx land:
    python paper_figure_generators_v2/build_v2_cache.py --metrics-root /storage_1/metrics_postfix \\
        --experiments "4000-4179,5000-5179,6000-6179"

    python compute_wilcoxon.py                 # tables + paste-ready LaTeX
    python compute_wilcoxon.py --patch --verde-repo ../VERDE   # patch \\pending{} in place

The --patch mode edits, in <verde-repo>/ieee_TSC_revisions/:
    main.tex, main_supplement.tex      -- replaces the \\pending{[pending rerun: ...]} box
    response.tex, response_supplement.tex -- replaces the '[pending rerun: ...]' excerpt
                                             placeholder and the 'pending a re-run' note
Run a LaTeX build afterwards and eyeball the R2C91 paragraph.
"""

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("ERROR: scipy not installed. On DRAC/huron: source ~/envs/merl_env/bin/activate",
          file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent
DEFAULT_CACHE = REPO / "paper_figure_generators_v2" / "table_data" / "_v2_cache" / "reward_per_exp.csv"

# Display names as used in the paper (alphabetical figure convention).
DISPLAY = {"CMA": "CMA-ES", "DQN": "DQN", "ODT": "ODT", "REINFORCE": "REINFORCE"}
# Pair order for reporting: principal pair first.
PAIRS = [("ODT", "REINFORCE"), ("ODT", "DQN"), ("DQN", "REINFORCE"),
         ("CMA", "ODT"), ("CMA", "DQN"), ("CMA", "REINFORCE")]


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    """Journal-style p formatting."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def run_pairs(wide, unit_label):
    """Paired Wilcoxon over the columns of a units x DMs frame. Returns list of dicts."""
    rows = []
    for a, b in PAIRS:
        if a not in wide.columns or b not in wide.columns:
            continue
        d = wide[[a, b]].dropna()
        n = len(d)
        if n < 2 or (d[a] - d[b]).abs().sum() == 0:
            rows.append({"a": a, "b": b, "n": n, "p": float("nan"),
                         "mean_a": d[a].mean() if n else float("nan"),
                         "mean_b": d[b].mean() if n else float("nan")})
            continue
        try:
            _, p = wilcoxon(d[a], d[b], alternative="two-sided", method="exact")
        except (TypeError, ValueError):
            _, p = wilcoxon(d[a], d[b], alternative="two-sided")
        rows.append({"a": a, "b": b, "n": n, "p": float(p),
                     "mean_a": float(d[a].mean()), "mean_b": float(d[b].mean())})
    ps = [r["p"] for r in rows]
    if all(p == p for p in ps):  # no NaNs
        for r, padj in zip(rows, holm(ps)):
            r["p_holm"] = padj
    else:
        for r in rows:
            r["p_holm"] = float("nan")
    print(f"\nPaired Wilcoxon signed-rank (two-sided), unit = {unit_label}:")
    for r in rows:
        sig = "significant" if r["p"] < 0.05 else "n.s."
        sigh = "sig." if r.get("p_holm", 1) < 0.05 else "n.s."
        print(f"  {DISPLAY[r['a']]:<10} vs {DISPLAY[r['b']]:<10} n={r['n']:>3}  "
              f"p={r['p']:.4f} ({sig})  Holm p={r['p_holm']:.4f} ({sigh})  "
              f"means {r['mean_a']:.2f} / {r['mean_b']:.2f}")
    return rows


def results_sentence(rows, design):
    """Build the LaTeX results sentence(s) that replace the \\pending box."""
    def ptex(p):
        return "$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"

    def pair_txt(r):
        return f"{DISPLAY[r['a']]} vs.\\ {DISPLAY[r['b']]} {ptex(r['p'])}"

    sig = [r for r in rows if r["p"] < 0.05]
    ns = [r for r in rows if not (r["p"] < 0.05)]
    principal = next((r for r in rows if {r["a"], r["b"]} == {"ODT", "REINFORCE"}), None)

    cma_sig = [r for r in sig if "CMA" in (r["a"], r["b"])]
    other_sig = [r for r in sig if "CMA" not in (r["a"], r["b"])]

    if len(sig) == len(rows):
        worst = max(r["p"] for r in rows)
        body = (f"All six pairwise reward differences are significant "
                f"(two-sided $p \\leq {worst:.3f}$ in every comparison), including the "
                f"principal ODT--REINFORCE pair ({ptex(principal['p'])})")
    elif not sig:
        best = min(rows, key=lambda r: r["p"])
        body = (f"None of the six pairwise differences reaches significance at "
                f"$\\alpha = 0.05$ (smallest {ptex(best['p'])}, "
                f"{DISPLAY[best['a']]} vs.\\ {DISPLAY[best['b']]})")
    elif len(cma_sig) == 3 and not other_sig:
        worst = max(r["p"] for r in cma_sig)
        ns_txt = "; ".join(pair_txt(r) for r in ns)
        body = ("CMA-ES differs significantly from each of the other three DMs "
                f"($p \\leq {worst:.3f}$ in every pairwise comparison), whereas the "
                f"differences among ODT, DQN, and REINFORCE are not significant at "
                f"$\\alpha = 0.05$ ({ns_txt})")
    else:
        sig_txt = "; ".join(pair_txt(r) for r in sig)
        ns_txt = "; ".join(pair_txt(r) for r in ns)
        body = (f"The differences {sig_txt} are significant at $\\alpha = 0.05$, "
                f"whereas {ns_txt} are not")

    holm_sig = [r for r in rows if r.get("p_holm", 1) < 0.05]
    same = {(r["a"], r["b"]) for r in holm_sig} == {(r["a"], r["b"]) for r in sig}
    holm_txt = (" The same comparisons remain significant under a Holm--Bonferroni "
                "correction across the six pairs." if same else
                " Under a Holm--Bonferroni correction across the six pairs, only "
                + ("; ".join(pair_txt(r) for r in holm_sig) if holm_sig else "no comparison")
                + " remains significant.")

    unit_note = ""
    if design == "config":
        unit_note = (" Here each DM pair is matched on the (season, aggregation "
                     "level, seed) configuration rather than collapsed per seed, "
                     "which increases the number of paired observations without "
                     "mixing unmatched conditions.")
    return body + "." + holm_txt + unit_note


def patch_file(path, pattern, replacement, desc):
    txt = path.read_text(encoding="utf-8")
    new, nsub = re.subn(pattern, replacement.replace("\\", "\\\\"), txt, flags=re.DOTALL)
    if nsub == 0:
        print(f"  [SKIP] {path.name}: {desc} placeholder not found (already patched?)")
        return False
    path.write_text(new, encoding="utf-8")
    print(f"  [OK]   {path.name}: replaced {desc} ({nsub} site)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE),
                    help="reward_per_exp.csv from build_v2_cache.py")
    ap.add_argument("--patch", action="store_true",
                    help="Patch the \\pending{} placeholders in the VERDE repo")
    ap.add_argument("--verde-repo", default=str(REPO.parent / "VERDE"),
                    help="Path to the VERDE repo (default: sibling of this repo)")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(
            f"{cache} not found. Build it first (see module docstring); on huron:\n"
            "  python paper_figure_generators_v2/build_v2_cache.py "
            "--metrics-root /storage_1/metrics_postfix "
            '--experiments "4000-4179,5000-5179,6000-6179"')

    per_exp = pd.read_csv(cache)
    seeds = sorted(per_exp["seed"].unique())
    n_seeds = len(seeds)
    print(f"Loaded {len(per_exp)} experiments; algorithms="
          f"{sorted(per_exp['algorithm'].unique())}; {n_seeds} seeds: {seeds}")

    # ---- Table 3 reward column (pooled across experiments) -----------------
    pooled = (per_exp.groupby("algorithm")["reward_final"]
              .agg(["mean", "std", "count"]).reset_index())
    print("\nTable 3 (tab:resource_analysis) Reward column, pooled across experiments:")
    for _, r in pooled.sort_values("algorithm").iterrows():
        print(f"  {DISPLAY.get(r['algorithm'], r['algorithm']):<10} "
              f"${r['mean']:.2f} \\pm {r['std']:.2f}$   (n={int(r['count'])} exps)")

    # ---- Design 1: per-seed (paper wording) --------------------------------
    per_seed = (per_exp.groupby(["algorithm", "seed"])["reward_final"]
                .mean().unstack("algorithm"))
    print("\nPer-seed final rewards:")
    print(per_seed.round(2).to_string())
    min_p = 2.0 ** -(n_seeds - 1)
    print(f"\n[design 1] n={n_seeds} seeds -> smallest achievable two-sided p = {min_p:.4f}"
          + ("  [UNDERPOWERED: cannot reach alpha=0.05; use design 2 or wait for "
             "the remaining seed batches]" if n_seeds < 6 else ""))
    rows_seed = run_pairs(per_seed, f"per-seed final reward (n={n_seeds} seeds)")

    # ---- Design 2: per-configuration ---------------------------------------
    per_cfg = (per_exp.groupby(["algorithm", "season", "num_aggs", "seed"])["reward_final"]
               .mean().unstack("algorithm"))
    rows_cfg = run_pairs(per_cfg, f"matched (season, aggregation, seed) configuration "
                                  f"(n={len(per_cfg)} configs)")

    # ---- Choose the design for the paper text ------------------------------
    if n_seeds >= 6:
        rows, design = rows_seed, "seed"
        design_txt = "the nine per-seed final rewards" if n_seeds == 9 else \
                     f"the {n_seeds} per-seed final rewards"
    else:
        rows, design = rows_cfg, "config"
        design_txt = "the final rewards of matched (season, aggregation, seed) configurations"
        print("\n[NOTE] Falling back to the per-configuration design for the LaTeX text; "
              "the manuscript sentence must then say 'matched configurations' instead of "
              "'nine per-seed final rewards'.")

    sentence = results_sentence(rows, design)

    print("\n" + "=" * 72)
    print("PASTE-READY LATEX (replaces the \\pending{...} box contents):")
    print("=" * 72)
    print(sentence)
    print("=" * 72)
    if design == "config":
        print("ALSO change the test-description in the same paragraph to:")
        print(f"  ... {design_txt} of each DM pair were compared with a paired "
              "Wilcoxon signed-rank test.")

    if not args.patch:
        print("\n(Re-run with --patch --verde-repo <path> to apply in place.)")
        return

    # ---- Patch the VERDE repo ----------------------------------------------
    tex_dir = Path(args.verde_repo) / "ieee_TSC_revisions"
    if not tex_dir.is_dir():
        raise SystemExit(f"--verde-repo: {tex_dir} not found")
    print(f"\nPatching placeholders under {tex_dir} ...")

    pending_re = r"\\pending\{\[pending rerun:.*?\]\}"
    for name in ("main.tex", "main_supplement.tex"):
        p = tex_dir / name
        if p.exists():
            patch_file(p, pending_re, sentence, "\\pending box")

    excerpt_re = r"\\textit\{\[pending rerun:.*?\]\}"
    note_re = r"\\emph\{Per-pair \$p\$-values are pending a re-run\.\}"
    n_sig = sum(1 for r in rows if r["p"] < 0.05)
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "all six"}
    note_replacement = (f"Of the six pairwise comparisons, {words.get(n_sig, n_sig)} "
                        f"{'is' if n_sig == 1 else 'are'} significant at "
                        f"$\\alpha=0.05$; see the excerpt for the per-pair $p$-values.")
    for name in ("response.tex", "response_supplement.tex"):
        p = tex_dir / name
        if p.exists():
            patch_file(p, excerpt_re, sentence, "excerpt placeholder")
            patch_file(p, note_re, note_replacement, "'pending a re-run' note")

    print("\nDone. Rebuild the PDFs and check the R2C91 paragraph, then update the "
          "Table 3 / Table 1 numbers if the 9-seed means moved (see "
          "paper_figure_generators_v2/Experiment 3 Figures.ipynb cell 16).")


if __name__ == "__main__":
    main()
