"""
reddit_scraper.py
CMPU 395 - Spring 2026
Aneesh Koppolu & Aspen Wang

Scrapes Reddit posts for each sampled school district using Reddit's
public JSON search (no API key required), then runs RoBERTa sentiment
analysis and correlates results with SEDA achievement scores.

Output files:
  - reddit_raw.csv         : raw scraped posts per district
  - reddit_sentiment.csv   : per-district aggregated sentiment scores
  - correlation_results.txt: Pearson correlations with SEDA scores
  - sentiment_vs_achievement.png : scatter plot
"""

import requests
import pandas as pd
import time
import json
import re
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# ── 1. LOAD DISTRICT DATA ────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH / "sampled_districts.csv")

# Keep one row per district (drop duplicate years if any)
df = df.drop_duplicates(subset=["sedalea"]).reset_index(drop=True)

print(f"Loaded {len(df)} districts")
print(df[["sedalea", "sedaleaname", "stateabb", "cs_mn_avg_eb"]].head())


# ── 2. REDDIT SCRAPER (no API key) ──────────────────────────────────────────

HEADERS = {"User-Agent": "CMPU395-ML-Project/1.0 (academic research)"}

def clean_text(text):
    if not text or str(text).strip() in ["", "[deleted]", "[removed]"]:
        return ""
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def search_reddit(query, max_posts=25):
    """Search Reddit public JSON endpoint for posts matching query."""
    url = "https://www.reddit.com/search.json"
    params = {
        "q": query,
        "sort": "relevance",
        "limit": max_posts,
        "t": "all",
        "type": "link"
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code == 429:
            print("  Rate limited — waiting 30s...")
            time.sleep(30)
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        posts = []
        for child in data["data"]["children"]:
            p = child["data"]
            title = clean_text(p.get("title", ""))
            selftext = clean_text(p.get("selftext", ""))
            combined = (title + " " + selftext).strip()
            if combined:
                posts.append({
                    "title": title,
                    "text": selftext,
                    "combined": combined,
                    "score": p.get("score", 0),
                    "subreddit": p.get("subreddit", ""),
                    "num_comments": p.get("num_comments", 0),
                })
        return posts
    except Exception as e:
        print(f"  Error: {e}")
        return []

def get_district_posts(district_name, state_abbr, max_posts=25):
    """Try multiple query strategies to maximize Reddit coverage."""
    # Clean district name - remove common suffixes for better search
    clean_name = re.sub(
        r"\b(unified school district|school district|union elementary|"
        r"elementary|cusd|usd|cooperative|community)\b",
        "", district_name, flags=re.IGNORECASE
    ).strip().rstrip("-,")

    queries = [
        f"{clean_name} {state_abbr} schools education",
        f"{clean_name} {state_abbr} school district",
        f"{clean_name} {state_abbr}",
    ]

    all_posts = []
    seen_titles = set()
    for query in queries:
        posts = search_reddit(query, max_posts=max_posts)
        for p in posts:
            if p["title"] not in seen_titles and len(p["combined"]) > 20:
                seen_titles.add(p["title"])
                all_posts.append(p)
        if len(all_posts) >= max_posts:
            break
        time.sleep(1.5)  # be polite to Reddit's servers

    return all_posts


# ── 3. SCRAPE ALL DISTRICTS ──────────────────────────────────────────────────

all_rows = []
failed_districts = []

for i, row in df.iterrows():
    # if i >=5:
    #     break
    lea_id = row["sedalea"]
    name = row["sedaleaname"]
    state = row["stateabb"]
    score = row["cs_mn_avg_eb"]

    print(f"[{i+1}/{len(df)}] {name}, {state} ...", end=" ", flush=True)

    posts = get_district_posts(name, state)

    if posts:
        print(f"{len(posts)} posts found")
        for p in posts:
            all_rows.append({
                "sedalea": lea_id,
                "sedaleaname": name,
                "stateabb": state,
                "cs_mn_avg_eb": score,
                **p
            })
    else:
        print("0 posts (no data)")
        failed_districts.append(name)

    time.sleep(2)  # ~2s between districts to avoid rate limiting

# Save raw data
raw_df = pd.DataFrame(all_rows)
raw_df.to_csv(DATA_PATH / "reddit_raw.csv", index=False)
print(f"\nSaved {len(raw_df)} total posts across {raw_df['sedalea'].nunique()} districts")
print(f"Districts with no Reddit data: {len(failed_districts)}")


# ── 4. ROBERTA SENTIMENT ANALYSIS ───────────────────────────────────────────

print("\nRunning RoBERTa sentiment analysis...")
print("Installing transformers if needed...")

import subprocess
subprocess.run(["pip", "install", "transformers", "torch", "--quiet"], check=False)

from transformers import pipeline

# Use a lightweight RoBERTa model fine-tuned on sentiment
# cardiffnlp/twitter-roberta-base-sentiment is compact and education-appropriate
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512
)

LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def score_text(text):
    """Return a numeric sentiment score: positive=1, neutral=0, negative=-1."""
    try:
        if not text or len(text.strip()) < 10:
            return None
        result = sentiment_pipe(text[:512])[0]
        label = result["label"].lower()
        confidence = result["score"]
        # Weight by confidence: e.g. 0.9 confident positive → ~0.9
        base = LABEL_MAP.get(label, 0.0)
        return base * confidence
    except Exception:
        return None

# Score each post
print("Scoring posts (this may take a few minutes)...")
raw_df["sentiment_score"] = raw_df["combined"].apply(score_text)
raw_df = raw_df.dropna(subset=["sentiment_score"])

# Aggregate to district level: mean sentiment score
sentiment_df = (
    raw_df.groupby(["sedalea", "sedaleaname", "stateabb", "cs_mn_avg_eb"])
    .agg(
        mean_sentiment=("sentiment_score", "mean"),
        post_count=("sentiment_score", "count"),
        std_sentiment=("sentiment_score", "std"),
    )
    .reset_index()
)

sentiment_df.to_csv(DATA_PATH / "reddit_sentiment.csv", index=False)
print(f"\nSentiment scores computed for {len(sentiment_df)} districts")
print(sentiment_df[["sedaleaname", "mean_sentiment", "post_count", "cs_mn_avg_eb"]].head(10))


# ── 5. CORRELATION ANALYSIS ──────────────────────────────────────────────────

# Filter to districts with enough posts for reliable sentiment
MIN_POSTS = 3
analysis_df = sentiment_df[sentiment_df["post_count"] >= MIN_POSTS].copy()
print(f"\nDistricts with >= {MIN_POSTS} posts for correlation: {len(analysis_df)}")

if len(analysis_df) >= 10:
    r, p = stats.pearsonr(analysis_df["mean_sentiment"], analysis_df["cs_mn_avg_eb"])
    spearman_r, spearman_p = stats.spearmanr(analysis_df["mean_sentiment"], analysis_df["cs_mn_avg_eb"])

    results_text = f"""
CORRELATION RESULTS — Reddit Sentiment vs. SEDA Achievement Scores
===================================================================
Districts analyzed: {len(analysis_df)}
Minimum posts per district: {MIN_POSTS}

Pearson Correlation:
  r = {r:.4f}
  p = {p:.4f}
  {'Statistically significant (p < 0.05)' if p < 0.05 else 'Not statistically significant (p >= 0.05)'}

Spearman Correlation:
  rho = {spearman_r:.4f}
  p   = {spearman_p:.4f}
  {'Statistically significant (p < 0.05)' if spearman_p < 0.05 else 'Not statistically significant (p >= 0.05)'}

Sentiment Score Summary:
  Mean:   {analysis_df['mean_sentiment'].mean():.4f}
  Std:    {analysis_df['mean_sentiment'].std():.4f}
  Min:    {analysis_df['mean_sentiment'].min():.4f}
  Max:    {analysis_df['mean_sentiment'].max():.4f}

SEDA Achievement Score Summary:
  Mean:   {analysis_df['cs_mn_avg_eb'].mean():.4f}
  Std:    {analysis_df['cs_mn_avg_eb'].std():.4f}
  Min:    {analysis_df['cs_mn_avg_eb'].min():.4f}
  Max:    {analysis_df['cs_mn_avg_eb'].max():.4f}
"""
    print(results_text)

    with open(DATA_PATH / "correlation_results.txt", "w") as f:
        f.write(results_text)

    # ── 6. SCATTER PLOT ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        analysis_df["mean_sentiment"],
        analysis_df["cs_mn_avg_eb"],
        alpha=0.6, edgecolors="white", linewidth=0.5, s=60, color="#2E5090"
    )

    # Regression line
    m, b = pd.Series(analysis_df["mean_sentiment"]).pipe(
        lambda x: (stats.linregress(x, analysis_df["cs_mn_avg_eb"])[:2])
    )
    x_range = pd.Series([analysis_df["mean_sentiment"].min(), analysis_df["mean_sentiment"].max()])
    ax.plot(x_range, m * x_range + b, color="#C0392B", linewidth=1.8, linestyle="--", label=f"r = {r:.3f}, p = {p:.3f}")

    ax.set_xlabel("Mean Reddit Sentiment Score", fontsize=12)
    ax.set_ylabel("SEDA Average Achievement Score", fontsize=12)
    ax.set_title("Reddit Sentiment vs. District Educational Achievement", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DATA_PATH / "sentiment_vs_achievement.png", dpi=150)
    print("Saved scatter plot to data/sentiment_vs_achievement.png")

else:
    print("Not enough districts with sufficient posts for correlation analysis.")
    print("Consider lowering MIN_POSTS threshold or expanding district sample.")

print("\nDone! Files saved to data/")
print("  - reddit_raw.csv")
print("  - reddit_sentiment.csv")
print("  - correlation_results.txt")
print("  - sentiment_vs_achievement.png")
