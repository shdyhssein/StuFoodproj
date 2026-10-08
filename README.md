GenAI Creative ROI: is it worth producing ad creatives with GenAI?

A small end-to-end study for a fictional student meal-prep app. A Python pipeline generates ad copy and image prompts with an LLM and logs cost, tokens and latency per asset. A rating survey (n = 14) tests 7 ads, a cost model compares GenAI with manual production, and an LLM judge's ratings are compared with the human ones.

Author: Shady Shiha, October 2026

Results in 4 lines
Cost: about €5 per usable asset with GenAI versus €25 / €50 / €100 for manual production taking 30 / 60 / 120 minutes (at €50/h). Break-even is after roughly 13 / 6 / 3 usable assets. Manual times are assumptions, not measurements.
Quality: 5 of 7 ads were rated well (mean quality 3.4-4.4), 2 poorly (1.6-1.9). The gap between good and poor ads of the same creative angle was larger than any gap between angles.
LLM judge: Claude Sonnet 5.5 agreed only weakly with the human ratings (Spearman 0.37-0.74 across the 7 ads) and did not flag the two weak ads. Exploratory, n = 7.
Implication: GenAI makes candidate creatives cheap, but human selection and a quick pre-test decide which ones are usable.

Full write-up: docs/report.pdf. One-slide summary: docs/slide.pdf.

How it works
config.yaml (brief, angles, models, prices)
        |
        v
src/generate.py --> LLM (Gemini text model) --> headline, body, image prompt
        |                                              |
        v                                              v
data/runs.csv (cost, tokens, latency, retries)   image made in a chat tool (manual step)
        |
        v
survey (7 ads, 14 raters) --> src/analyze_ratings.py --> data/human_means.csv
        |                                                      |
        v                                                      v
src/cost_model.py --> scenarios,              src/compare_llm.py (LLM judge vs humans)
break-even, Monte Carlo, charts
Repository structure
src/
  generate.py          generates assets and logs one row per asset
  analyze_ratings.py   survey statistics and chart
  cost_model.py        cost per usable asset, break-even, scenarios
  compare_llm.py       compares the LLM judge's ratings with the human ratings
data/
  runs.csv             log of generated assets (cost, tokens, latency, image time)
  ratings_long.csv     survey ratings, one row per respondent and ad
  llm_ratings.csv      LLM judge ratings (4 runs, 7 ads)
  survey_key.csv       maps asset IDs to creative angle and headline
  human_means.csv      mean ratings per ad (output of analyze_ratings.py)
  cost_scenarios.csv   scenario grid (output of cost_model.py)
docs/
  report.pdf, slide.pdf, survey.md
  ads/                 the 7 ads shown in the survey (asset01.png to asset07.png)
  ratings_by_ad.png, breakeven_cumulative.png, breakeven_heatmap.png, llm_vs_human.png
config.yaml            brief, creative angles, models, prices, hourly rates
.env.example           name of the API key variable (copy to .env)
requirements.txt
.gitignore             keeps .env, .venv and local files out of the repo
Setup

Requires Python 3.11 or newer.

python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
Reproduce the analysis (no API key needed)
python src/analyze_ratings.py   # survey statistics, docs/ratings_by_ad.png, data/human_means.csv
python src/cost_model.py        # data/cost_scenarios.csv, docs/breakeven_*.png
python src/compare_llm.py       # LLM judge vs humans, docs/llm_vs_human.png

Run analyze_ratings.py first, because the other two scripts read data/human_means.csv.

Generate new assets (needs a Gemini API key)
Copy .env.example to .env and set GEMINI_API_KEY=your_key. Never commit .env.
Check model names and prices in config.yaml, since they change often.
Run python src/generate.py --n 3 for a test, then a larger batch.

Each run appends rows to data/runs.csv. Image time, usability and review time are filled in by hand. All durations are in decimal minutes (4.5 = 4 min 30 s).

Data notes
ratings_long.csv: version 1 = ads shown in order 1-7, version 2 = reversed. Ratings are 1-5 (quality, fit, click intent). ai_look = whether the respondent thought the ad looked AI-made. Responses are anonymous, and field of study was removed.
llm_ratings.csv: Claude Sonnet 5.5 rated the same 7 ads (images, headline and text, all in one message) in 4 separate runs with the same three 1-5 scales and the same "looks AI-made" question. The ad order differed between some runs.
The 7 ads shown in the survey are in docs/ads/, and their image prompts are in data/runs.csv. docs/survey.md describes the survey questions.
Assumptions and limitations
Assumed, not measured: manual production time per asset (30 / 60 / 120 min), 2 min prompt and selection time per attempt, 5 h setup time, hourly rates (€15 / €50 / €100).
Modeled: image cost uses a list price (about €0.03 per image). Images were created in a chat tool, so only that step's time was logged.
Usable rate (71%) is a proxy: 5 of 7 tested ads had a mean quality of at least 3. The 7 ads were selected from 30 candidates.
Survey: small convenience sample (friends and university groups), exploratory statistics only. The product name was part of the brief but not required in every ad text, so several ads say "StuFood" and others do not. Ad 3 differs in image style.
LLM judge: only 7 ads, so correlations are rough. The judge saw all 7 ads together in one message, unlike the human respondents, and it belongs to the same model family (Claude) that helped write the code and report, so a self-preference bias cannot be excluded.
Relative savings barely depend on the hourly rate, because manual cost, GenAI cost and setup cost all scale with it.
Governance notes

All generated assets carry an ai_label flag in the log. AI-generated content may need to be labeled under EU AI Act transparency rules, so check the current scope and timing. This is not legal advice. Prompts exclude real people's likenesses, and ads are reviewed by a human before use.

AI assistance

The code, report and README were drafted with Claude Sonnet 5.5 (Anthropic) assistance. I defined the study, ran the pipeline, created the images, collected the survey data, and reviewed, tested and adapted the code and text. The LLM judge in the comparison was also Claude Sonnet 5.5.