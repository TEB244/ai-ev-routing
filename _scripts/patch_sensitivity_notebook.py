"""
Append a 'Reward-weight recommendation' section to
paper_figure_generators/Sensitivity Analysis.ipynb.

This is idempotent - if the section is already present (detected by a
marker string in the markdown), running again will replace it rather
than duplicate. Existing executed outputs of unrelated cells are
preserved.

Run from the repo root:
    python _scripts/patch_sensitivity_notebook.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH   = REPO_ROOT / "paper_figure_generators" / "Sensitivity Analysis.ipynb"

MARKER = "# Part 3 - Recommended reward weights for the rerun"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }


def split_lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build_section_cells():
    cells = []
    cells.append(md(*split_lines("""# Part 3 - Recommended reward weights for the rerun

This section consolidates the sensitivity-sweep evidence into a single
recommended reward parameterisation to apply to all 4xxx, 5xxx, and
6xxx main experiments when rerunning them.

## Recommendation

```yaml
environment_settings:
  # Unit-conversion scales (fixed)
  distance_scale: 100.0    # deg lat/long -> approximate km
  traffic_scale:  1.0      # raw cars at peak station
  energy_scale:  0.001     # Wh -> kWh

  # Tunable per-term weights (recommended values)
  distance_weight: 1.0
  traffic_weight:  1.0
  energy_weight:   1.0
```

## Rationale

The recommendation rests on three findings from the 7xxx sweep:

**1. DM ranking is preserved across the full {0, 1, 5, 7, 10} sweep
   for every weight.** The ranking is identical at every cell except
   `traffic_weight = 0` where ablating the dominant penalty causes a
   three-way collapse to ~-40 reward. Since rankings are invariant to
   weight choice within the tested range, any value in [1, 10] yields
   the same qualitative conclusions; we pick the unit value as the
   minimal, least-arbitrary choice.

**2. The scales act as physical unit conversions, the weights act as
   relative importance.** With `distance_scale = 100`, `distance` in
   degrees lat/long times the scale yields approximately km. Similarly
   `traffic_scale = 1` keeps the raw count of cars at the peak station,
   and `energy_scale = 0.001` converts Wh to kWh. Holding weights at 1
   means the reward is a literal sum of penalties in commensurate
   natural units (km of travel + cars at peak + kWh charged), which is
   the most defensible and reviewer-resistant choice. Any departure
   from `weight = 1` requires justifying why one natural-unit penalty
   should count more than another.

**3. Agents demonstrably respond to the weighted reward.** Figure B
   shows distance_mean and mean_peak_traf change between models trained
   under different reward configurations - DQN converges to ~9-10 cars
   peak traffic vs CMA's ~22, and DQN's distance is consistently
   shorter. This rules out the alternative interpretation that the
   reward shape is irrelevant.

## What this changes vs the original paper

The original paper used scales `(100, 1, 0.001)` and implicit weights
of `(1, 1, 1)` - meaning the recommendation is **the same** as what
the original paper used, but is now explicitly defended rather than
asserted by code defaults. The paper text should:

1. Add explicit values for the weights and scales to the methods
   section, near Eq. 5.
2. Cite the sensitivity analysis appendix as the justification.
3. Replace the hand-wave \"normalisation weights that balance the
   influence of each factor\" with the unit-conversion framing.

## Quantitative decomposition at the recommended values

At the recommended `(1, 1, 1)` weights, the per-episode reward
decomposes (post-fix DQN agent, from the sensitivity sweep slope
analysis) into:""")))

    cells.append(code(*split_lines("""# Reward decomposition at the recommended (1, 1, 1) weights, from
# the 7xxx sweep slope analysis. The slope of reward vs weight along
# a given axis IS the per-episode contribution of that term at
# baseline (because behaviour is essentially weight-invariant within
# the sweep range, so reward changes are purely the multiplier on the
# fixed per-episode quantity).

import pandas as pd
from pathlib import Path

p = Path('table_data') / 'sensitivity_summary.csv'
if not p.exists():
    print('sensitivity_summary.csv not found - run Parts 1 and 2 first.')
else:
    df = pd.read_csv(p)
    # Extract mean values per (model, swept_var, weight)
    def parse_mean(s):
        try:
            return float(s.split('+/-')[0].strip())
        except Exception:
            return float('nan')

    rows = []
    for model in df['model'].unique():
        for var in ['distance_weight', 'traffic_weight', 'energy_weight']:
            sub = df[(df['model'] == model) & (df['swept_var'] == var)]
            if sub.empty:
                continue
            r1  = parse_mean(sub.iloc[0]['1'])
            r10 = parse_mean(sub.iloc[0]['10'])
            slope = (r10 - r1) / 9.0
            rows.append({
                'model':      model,
                'term':       var.replace('_weight', ''),
                'contribution_per_episode': abs(slope),
            })
    decomp = pd.DataFrame(rows).pivot(index='model', columns='term', values='contribution_per_episode').round(2)
    decomp['total |R|'] = decomp.sum(axis=1).round(2)
    print('Per-episode reward contribution at recommended (1, 1, 1) weights:')
    display(decomp)
    print()
    print('Interpretation:')
    print('  - Distance contribution is ~36-37 units per episode for all DMs')
    print('    (~36-37 km of travel).')
    print('  - Traffic contribution varies meaningfully by DM:')
    print('    DQN ~39 (best), REINFORCE ~49, CMA ~74. This is the gap')
    print('    that drives the DM ranking.')
    print('  - Energy contribution is small (~3.5-4 kWh-equivalent),')
    print('    serves as a soft regularizer.')""")))

    cells.append(md(*split_lines("""## Why not other weight values?

A few alternatives we considered and rejected:

* **`weight = 0` ablation cells** - useful as a sensitivity probe,
  not as a default. Ablating any single term changes the optimisation
  target qualitatively.
* **Up-weighting traffic (e.g., `traffic_weight = 5`)** to emphasise
  the dominant DM-differentiation signal - rejected because reward
  magnitudes blow up to several hundred without changing behaviour
  much, making cross-paper comparisons harder.
* **Rebalancing to equalise the per-episode contributions** (e.g.,
  raise `energy_weight` to ~10 so energy is on par with distance and
  traffic) - tempting but introduces a calibration step that depends
  on the agent's policy, which is itself what we are training. The
  unit-conversion framing avoids that circularity.

## Action items for the rerun

1. Apply the recommended snippet above to all 4xxx, 5xxx, and 6xxx
   `environment_settings` blocks.
2. The 7xxx and 8xxx configs already use these values (post-fix).
3. Cite the sensitivity analysis appendix (this notebook + Figures A,
   B, C) as the empirical justification in the paper revision.""")))

    return cells


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Drop any pre-existing Part 3 (idempotent)
    keep = []
    skipping = False
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and MARKER in src:
            skipping = True
            continue
        if skipping:
            # Stop dropping when we hit the next top-level header
            if c["cell_type"] == "markdown" and src.lstrip().startswith("# ") and MARKER not in src:
                skipping = False
                keep.append(c)
            # otherwise drop
            continue
        keep.append(c)

    keep.extend(build_section_cells())
    nb["cells"] = keep

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH}  ({len(keep)} cells total)")


if __name__ == "__main__":
    main()
