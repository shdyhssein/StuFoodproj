import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import os
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

FIELDS = [
    "asset_id", "timestamp", "prompt_version", "text_model", "image_model_planned",
    "angle", "input_tokens", "output_tokens", "text_cost_eur", "image_cost_eur_modeled",
    "latency_s", "retries", "headline", "body", "image_prompt",
    # filled in by hand later:
    "image_file", "image_gen_min", "usable", "human_review_min",
    "ai_label",
]

PROMPT_TEMPLATE = """You are a creative copywriter for a marketing team.

Product: {product}
Target audience: {audience}
Tone: {tone}
Key message: {key_message}
Creative angle for this ad: {angle}

Write ONE ad. Return ONLY valid JSON with exactly these keys:
- "headline": max 8 words
- "body": 1-2 sentences, max 30 words
- "image_prompt": a detailed prompt for an image generator describing one ad visual
  (scene, style, colors, composition). The image must contain no real people's likenesses or copyright infringement, do not immitate existing ads.
"""


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def next_asset_id(csv_path):
    if not csv_path.exists():
        return 1
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows) + 1


def call_model(client, cfg, prompt):
    """Call the text model with retries on server/quota errors. Returns (response, retries, latency_s)."""
    retries = 0
    start = time.time()
    while True:
        try:
            response = client.models.generate_content(
                model=cfg["text_model"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=cfg["temperature"],
                    response_mime_type="application/json",
                ),
            )
            return response, retries, time.time() - start
        except errors.APIError as e:
            if e.code in (429, 500, 503) and retries < cfg["max_retries"]:
                wait = 3 * 2 ** retries
                retries += 1
                print(f"  API busy ({e.code}), retry {retries} in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="number of assets to generate")
    args = parser.parse_args()

    cfg = load_config()
    if args.n > cfg["max_assets_per_run"]:
        raise SystemExit(f"--n {args.n} exceeds max_assets_per_run ({cfg['max_assets_per_run']}).")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found. Check your .env file in the project root.")
    client = genai.Client(api_key=api_key)

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "runs.csv"
    is_new = not csv_path.exists()
    asset_id = next_asset_id(csv_path)

    p = cfg["prices_usd"]
    fx = cfg["eur_per_usd"]
    image_cost_eur = p["image_per_image"] * fx
    b = cfg["brief"]
    angles = cfg["angles"]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        total_cost = 0.0
        for i in range(args.n):
            angle = angles[i % len(angles)]
            prompt = PROMPT_TEMPLATE.format(angle=angle, **b)
            print(f"Asset {asset_id} ({angle.split(':')[0]})...")

            try:
                response, retries, latency = call_model(client, cfg, prompt)
                asset = parse_json(response.text)
            except Exception as e:
                print(f"  FAILED: {str(e)[:120]}")
                continue

            usage = response.usage_metadata
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = (getattr(usage, "candidates_token_count", 0) or 0) + \
                      (getattr(usage, "thoughts_token_count", 0) or 0)
            text_cost_eur = (in_tok / 1e6 * p["text_input_per_1m_tokens"]
                             + out_tok / 1e6 * p["text_output_per_1m_tokens"]) * fx
            total_cost += text_cost_eur

            writer.writerow({
                "asset_id": asset_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt_version": cfg["prompt_version"],
                "text_model": cfg["text_model"],
                "image_model_planned": cfg["image_model_planned"],
                "angle": angle,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "text_cost_eur": round(text_cost_eur, 6),
                "image_cost_eur_modeled": round(image_cost_eur, 4),
                "latency_s": round(latency, 2),
                "retries": retries,
                "headline": asset.get("headline", ""),
                "body": asset.get("body", ""),
                "image_prompt": asset.get("image_prompt", ""),
                "image_file": "", "image_gen_min": "", "usable": "", "human_review_min": "",
                "ai_label": "true",
            })
            f.flush()
            print(f"  {asset.get('headline', '')}  | {latency:.1f}s, {retries} retries, EUR {text_cost_eur:.5f}")
            asset_id += 1

    print(f"\nDone. Text cost this run: EUR {total_cost:.5f}. Log: {csv_path}")


if __name__ == "__main__":
    main()