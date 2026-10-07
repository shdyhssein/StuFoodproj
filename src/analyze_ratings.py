"""
Reads data/ratings_long.csv (one row per respondent x ad) and prints descriptive
statistics, a Friedman test across ads and an order check. Saves a chart
(docs/ratings_by_ad.png) and data/human_means.csv (used later to compare with
the LLM judge).

Columns in ratings_long.csv:
  respondent  anonymous ID (R1..R14)
  version     1 = ads shown in order 1..7, 2 = reversed order 7..1
  asset_id    which ad (1-7, see data/survey_key.csv)
  quality, fit, click   ratings from 1 to 5
  ai_look     "yes" / "not sure" / "no" (does the ad look AI-made?)
  angle       creative angle of the ad
  ai_yes      1 if ai_look == "yes", else 0
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
DOCS.mkdir(exist_ok=True)
METRICS = ["quality", "fit", "click"]

df = pd.read_csv(DATA / "ratings_long.csv")
n_resp = df["respondent"].nunique()
print(f"{n_resp} respondents, {len(df)} ratings, per version: "
      f"{df.groupby('version')['respondent'].nunique().to_dict()}\n")

angle_of = df.drop_duplicates("asset_id").set_index("asset_id")["angle"]

#Means per ad
g = df.groupby("asset_id")
per_ad = g[METRICS].mean().round(2)
per_ad["ai_yes_rate"] = g["ai_yes"].mean().round(2)
per_ad["not_sure_rate"] = g["ai_look"].apply(lambda s: (s == "not sure").mean()).round(2)
print("--- Per ad ---")
print(per_ad, "\n")
out = per_ad.copy()
out.insert(0, "angle", angle_of)
out.reset_index().to_csv(DATA / "human_means.csv", index=False)

#Means per angle (confounded with how well each ad was executed
print("--- Per angle ---")
print(df.groupby("angle")[METRICS].mean().round(2), "\n")


def friedman(data, label):
    print(f"--- Friedman test across ads{label} ---")
    for m in METRICS:
        piv = data.pivot(index="respondent", columns="asset_id", values=m)
        stat, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
        w = stat / (len(piv) * (piv.shape[1] - 1))  # Kendall's W (effect size)
        print(f"{m:8s} chi2={stat:.2f}  p={p:.2g}  Kendall's W={w:.2f}")
    print()


#Do the ads differ?
friedman(df, "")

#Same test without the two weakest ads
weak = per_ad["quality"].nsmallest(2).index.tolist()
friedman(df[~df["asset_id"].isin(weak)], f" without the weakest ads {weak}")

#Order check
print("--- Order check: version 1 vs 2 (Mann-Whitney on per-person means) ---")
pr = df.groupby(["respondent", "version"])[METRICS].mean().reset_index()
for m in METRICS:
    a = pr.loc[pr["version"] == 1, m]
    b = pr.loc[pr["version"] == 2, m]
    u, p = stats.mannwhitneyu(a, b)
    print(f"{m:8s} v1={a.mean():.2f}  v2={b.mean():.2f}  p={p:.3f}")
print()

#Chart
means = df.groupby("asset_id")[METRICS].mean()
sems = df.groupby("asset_id")[METRICS].sem()
ax = means.plot(kind="bar", yerr=sems, capsize=3, figsize=(10, 5))
ax.set_ylim(1, 5)
ax.set_ylabel("Mean rating (1-5), error bars = SEM")
ax.set_xlabel("Ad (asset ID and creative angle)")
ax.set_xticklabels([f"{i}\n{angle_of[i]}" for i in means.index], rotation=0)
ax.set_title(f"Human ratings per ad (n={n_resp} respondents)")
plt.tight_layout()
plt.savefig(DOCS / "ratings_by_ad.png", dpi=200)
print("Saved docs/ratings_by_ad.png and data/human_means.csv")