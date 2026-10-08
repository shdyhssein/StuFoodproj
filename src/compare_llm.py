import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
DOCS.mkdir(exist_ok=True)
METRICS = ["quality", "fit", "click"]

llm_rows = pd.read_csv(DATA / "llm_ratings.csv")
human = pd.read_csv(DATA / "human_means.csv").set_index("asset_id")
llm = llm_rows.groupby("asset_id")[METRICS].mean()
n_runs = llm_rows["run"].nunique()
judge_name = ", ".join(llm_rows["judge"].unique())
print(f"Judge(s): {llm_rows['judge'].unique().tolist()}, {n_runs} runs, {llm.shape[0]} ads\n")

print("--- Mean rating per ad: LLM vs human ---")
table = pd.concat({"llm": llm.round(2), "human": human[METRICS].round(2)}, axis=1)
print(table, "\n")

print("--- Agreement across the ads (n = 7, exploratory) ---")
for m in METRICS:
    rho, p = stats.spearmanr(llm[m], human[m])
    mae = (llm[m] - human[m]).abs().mean()
    bias = (llm[m] - human[m]).mean()
    print(f"{m:8s} Spearman rho={rho:.2f} (p={p:.2f})  mean abs gap={mae:.2f}  "
          f"LLM minus human={bias:+.2f}  LLM range {llm[m].min():.2f}-{llm[m].max():.2f}  "
          f"human range {human[m].min():.2f}-{human[m].max():.2f}")
print()

print("--- Stability of the judge across runs ---")
for m in METRICS:
    piv = llm_rows.pivot(index="asset_id", columns="run", values=m)
    cors = [stats.spearmanr(piv[a], piv[b])[0] for a, b in itertools.combinations(piv.columns, 2)]
    sd = llm_rows.groupby("asset_id")[m].std().mean()
    print(f"{m:8s} mean run-to-run Spearman={np.nanmean(cors):.2f}  mean SD per ad={sd:.2f}")
print()

print("--- 'Looks AI-made' (share saying yes): LLM vs human ---")
llm_yes = llm_rows.groupby("asset_id")["ai_look"].apply(lambda s: (s == "yes").mean())
print(pd.DataFrame({"llm_yes": llm_yes.round(2), "human_yes": human["ai_yes_rate"].round(2)}), "\n")

# ---- Chart: human vs LLM, quality and click intent ----
lo = llm_rows.groupby("asset_id")[METRICS].min()
hi = llm_rows.groupby("asset_id")[METRICS].max()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
x = np.arange(len(llm))
for ax, m in zip(axes, ["quality", "click"]):
    ax.bar(x - 0.2, human[m], 0.4, label="Humans (n=14)")
    err = np.vstack([llm[m] - lo[m], hi[m] - llm[m]])
    ax.bar(x + 0.2, llm[m], 0.4, yerr=err, capsize=3, label=f"{judge_name} ({n_runs} runs, min-max)")
    ax.set_xticks(x, [str(i) for i in llm.index])
    ax.set_xlabel("Ad")
    ax.set_title(m.capitalize())
    ax.set_ylim(1, 5)
axes[0].set_ylabel("Mean rating (1-5)")
axes[0].legend(loc="lower left")
plt.tight_layout()
plt.savefig(DOCS / "llm_vs_human.png", dpi=200)
print("Saved docs/llm_vs_human.png")