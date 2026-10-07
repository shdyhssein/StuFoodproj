"""
Business case: when does GenAI creative production beat manual production?

REAL inputs (measured/logged): text cost, modeled image price and image-generation
time and review time from data/runs.csv; usable rate from the survey (data/human_means.csv).
ASSUMED inputs: manual minutes per asset, prompt/selection time, setup hours, hourly rates.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
DOCS.mkdir(exist_ok=True)

MANUAL_MIN = {"fast": 30, "base": 60, "slow": 120}
PROMPT_SELECT_MIN = 2.0
SETUP_HOURS = 5.0
MANUAL_TOOL_COST = 0.0
USABLE_LOW, USABLE_HIGH = 0.40, 1.00
USABLE_BASE_DEFAULT = 0.70
BASE_RATE_LABEL = "freelancer"

cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RATES = cfg["hourly_rates_eur"]

#Real inputs from the logs
runs = pd.read_csv(DATA / "runs.csv", skipinitialspace=True)
for c in ["text_cost_eur", "image_cost_eur_modeled", "image_gen_min", "human_review_min"]:
    runs[c] = pd.to_numeric(runs[c], errors="coerce")
api_eur = runs["text_cost_eur"].mean() + runs["image_cost_eur_modeled"].mean()
img_min = runs["image_gen_min"].mean()
rev_min = runs["human_review_min"].mean()
n_timed = runs["image_gen_min"].notna().sum()

#Usable rate: share of tested ads with mean quality >= 3 in the survey
hm_path = DATA / "human_means.csv"
if hm_path.exists():
    hm = pd.read_csv(hm_path)
    usable_base = float((hm["quality"] >= 3).mean())
    usable_src = f"{int((hm['quality'] >= 3).sum())} of {len(hm)} ads rated >= 3 on quality in the survey"
else:
    usable_base = USABLE_BASE_DEFAULT
    usable_src = "default assumption (no survey file found)"
USABLE = {"low": USABLE_LOW, "base": usable_base, "high": USABLE_HIGH}


def manual_cost(rate, minutes):
    return minutes / 60 * rate + MANUAL_TOOL_COST


def genai_cost(rate, usable, p_min=PROMPT_SELECT_MIN, total_min=None):
    """Cost per USABLE asset: every attempt pays text + image + human time; only a share is usable.
    total_min overrides the human minutes per attempt (used for the sensitivity chart)."""
    minutes = img_min + rev_min + p_min if total_min is None else total_min
    return (api_eur + minutes / 60 * rate) / usable


def breakeven(rate, minutes, usable, total_min=None):
    saving = manual_cost(rate, minutes) - genai_cost(rate, usable, total_min=total_min)
    return SETUP_HOURS * rate / saving if saving > 0 else np.inf


print("=== Inputs ===")
print(f"API cost per attempt (text + modeled image): EUR {api_eur:.4f}")
print(f"Image generation time: {img_min:.2f} min (mean of {n_timed} timed assets)")
print(f"Review time: {rev_min:.2f} min, prompt/selection time (assumed): {PROMPT_SELECT_MIN} min")
print(f"Usable rate base: {usable_base:.2f} ({usable_src})")
print(f"Setup (assumed): {SETUP_HOURS} h, hourly rates: {RATES}\n")

#Scenario grid
rows = []
for mlabel, mmin in MANUAL_MIN.items():
    for rlabel, rate in RATES.items():
        for ulabel, u in USABLE.items():
            m, g = manual_cost(rate, mmin), genai_cost(rate, u)
            rows.append(dict(manual_scenario=mlabel, manual_min=mmin, rate_label=rlabel, rate_eur_h=rate,
                             usable_scenario=ulabel, usable_rate=round(u, 2),
                             manual_eur=round(m, 2), genai_eur=round(g, 2),
                             saving_eur=round(m - g, 2), saving_pct=round(100 * (m - g) / m, 1),
                             breakeven_assets=round(breakeven(rate, mmin, u), 1)))
grid = pd.DataFrame(rows)
grid.to_csv(DATA / "cost_scenarios.csv", index=False)

print("=== Cost per usable asset, base usable rate (EUR) ===")
show = grid[grid["usable_scenario"] == "base"][
    ["manual_scenario", "manual_min", "rate_label", "manual_eur", "genai_eur", "saving_pct", "breakeven_assets"]]
print(show.to_string(index=False), "\n")

print("=== Sensitivity to usable rate (manual base, freelancer rate) ===")
print(grid[(grid["manual_scenario"] == "base") & (grid["rate_label"] == BASE_RATE_LABEL)][
    ["usable_scenario", "usable_rate", "manual_eur", "genai_eur", "saving_pct", "breakeven_assets"]
].to_string(index=False), "\n")

#Tipping point: how much human time per GenAI attempt erases the advantage?
print("=== Tipping point (human minutes per GenAI attempt at which GenAI stops being cheaper) ===")
rate_b = RATES[BASE_RATE_LABEL]
for mlabel, mmin in MANUAL_MIN.items():
    tip = 60 * (manual_cost(rate_b, mmin) * USABLE["base"] - api_eur) / rate_b
    print(f"manual {mmin:3d} min -> GenAI would need ~{tip:.0f} human min per attempt "
          f"(current estimate: {img_min + rev_min + PROMPT_SELECT_MIN:.1f})")
print("Note: relative savings barely depend on the hourly rate, because manual cost, GenAI cost "
      "and setup cost all scale with it. The rate changes the euro amounts, not the break-even volume.\n")

#Monte Carlo over the uncertain inputs (freelancer rate)
rng = np.random.default_rng(42)
N = 10_000
rate = RATES[BASE_RATE_LABEL]
mm = rng.triangular(MANUAL_MIN["fast"], MANUAL_MIN["base"], MANUAL_MIN["slow"], N)
uu = rng.uniform(USABLE_LOW, USABLE_HIGH, N)
pp = rng.uniform(1.0, 5.0, N)
manual_draw = mm / 60 * rate + MANUAL_TOOL_COST
genai_draw = (api_eur + (img_min + rev_min + pp) / 60 * rate) / uu
sav = 100 * (manual_draw - genai_draw) / manual_draw
print(f"=== Monte Carlo ({N} draws, rate EUR {rate}/h) ===")
print(f"Saving per asset: median {np.median(sav):.0f}%, 5th-95th percentile "
      f"{np.percentile(sav, 5):.0f}% to {np.percentile(sav, 95):.0f}%")
print(f"Share of draws where GenAI is cheaper per usable asset: {(sav > 0).mean():.0%}\n")

#Chart 1: cumulative cost, base case
rate = RATES[BASE_RATE_LABEL]
m_c, g_c = manual_cost(rate, MANUAL_MIN["base"]), genai_cost(rate, USABLE["base"])
be = breakeven(rate, MANUAL_MIN["base"], USABLE["base"])
nmax = int(max(30, np.ceil(2.5 * be))) if np.isfinite(be) else 30
x = np.arange(0, nmax + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, x * m_c, label=f"Manual ({MANUAL_MIN['base']} min/asset, assumed)")
ax.plot(x, SETUP_HOURS * rate + x * g_c, label=f"GenAI (setup EUR {SETUP_HOURS * rate:.0f} + EUR {g_c:.2f}/asset)")
if np.isfinite(be):
    ax.axvline(be, color="grey", linestyle="--")
    ax.text(be + 0.5, ax.get_ylim()[1] * 0.05, f"break-even ~{be:.0f} assets", color="grey")
ax.set_xlabel("Usable assets produced")
ax.set_ylabel("Cumulative cost (EUR)")
ax.set_title(f"Base case: EUR {rate}/h, usable rate {USABLE['base']:.0%}")
ax.legend()
plt.tight_layout()
plt.savefig(DOCS / "breakeven_cumulative.png", dpi=200)
plt.close()

#Chart 2: break-even heatmap (manual minutes x GenAI human minutes per attempt)
mins = [15, 30, 45, 60, 90, 120, 150, 180]
gmins = [5, 10, 15, 20, 30, 45]
rate_h = RATES[BASE_RATE_LABEL]
Z = np.array([[breakeven(rate_h, mi, USABLE["base"], total_min=g) for mi in mins] for g in gmins])
cap = 150
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(np.clip(np.where(np.isfinite(Z), Z, cap), 0, cap), cmap="viridis_r", aspect="auto", origin="lower")
ax.set_xticks(range(len(mins)), labels=[str(m) for m in mins])
ax.set_yticks(range(len(gmins)), labels=[str(g) for g in gmins])
for i in range(len(gmins)):
    for j in range(len(mins)):
        v = Z[i, j]
        txt = "never" if not np.isfinite(v) else (f">{cap}" if v > cap else f"{v:.0f}")
        dark = (not np.isfinite(v)) or v > cap * 0.45
        ax.text(j, i, txt, ha="center", va="center", color="white" if dark else "black", fontsize=9)
ax.set_xlabel("Manual minutes per asset (assumed)")
ax.set_ylabel("Human minutes per GenAI attempt")
ax.set_title(f"Break-even volume (usable assets), usable rate {USABLE['base']:.0%}, EUR {rate_h}/h")
plt.colorbar(im, label="assets (capped)")
plt.tight_layout()
plt.savefig(DOCS / "breakeven_heatmap.png", dpi=200)
plt.close()

print("Saved data/cost_scenarios.csv, docs/breakeven_cumulative.png, docs/breakeven_heatmap.png")