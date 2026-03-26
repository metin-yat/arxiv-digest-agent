"""
ArXiv Research Digest Agent
============================
Fetches recent papers from ArXiv, scores them for relevance to 3D Computer Vision
and NeRF using Gemini 2.5 Flash, summarizes top papers, and sends a formatted
HTML digest email via Gmail SMTP.
"""

import os
import sys
import time
import logging
import smtplib
import textwrap
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import feedparser
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
GMAIL_ADDRESS    = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

ARXIV_BASE_URL   = "http://export.arxiv.org/api/query"
KEYWORDS         = ["Gaussian Splatting", "NeRF", "Computer Vision"]
RELEVANCE_THRESHOLD = 6          # Papers scoring below this are filtered out
MAX_RESULTS_PER_KEYWORD = 30     # ArXiv results to fetch per keyword
GEMINI_MODEL     = "gemini-2.5-flash"
SCORE_RETRIES    = 2             # Retry attempts for Gemini API calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ArXiv Fetching
# ---------------------------------------------------------------------------

def build_arxiv_query(keyword: str) -> str:
    """Build an ArXiv API query string for the given keyword."""
    return f'ti:"{keyword}" OR abs:"{keyword}"'


def fetch_arxiv_papers(keyword: str, max_results: int = MAX_RESULTS_PER_KEYWORD) -> list[dict]:
    """
    Fetch papers from ArXiv matching *keyword* published in the last 24 hours.
    Returns a list of dicts with keys: id, title, abstract, authors, published, url.
    """
    params = {
        "search_query": build_arxiv_query(keyword),
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    log.info("Fetching ArXiv papers for keyword: '%s'", keyword)
    try:
        response = requests.get(ARXIV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.error("ArXiv request failed for '%s': %s", keyword, exc)
        return []

    feed = feedparser.parse(response.text)
    papers = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    for entry in feed.entries:
        # Parse published date
        published_str = entry.get("published", "")
        try:
            published_dt = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            published_dt = datetime.now(timezone.utc)

        if published_dt < cutoff:
            continue  # Skip papers older than 24 hours

        paper = {
            "id": entry.get("id", ""),
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "abstract": entry.get("summary", "").replace("\n", " ").strip(),
            "authors": [a.get("name", "") for a in entry.get("authors", [])],
            "published": published_str,
            "url": entry.get("id", ""),
        }
        papers.append(paper)

    log.info("  Found %d papers in the last 24h for '%s'", len(papers), keyword)
    return papers


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by ArXiv ID."""
    seen = set()
    unique = []
    for paper in papers:
        pid = paper["id"]
        if pid not in seen:
            seen.add(pid)
            unique.append(paper)
    log.info("Deduplicated to %d unique papers total.", len(unique))
    return unique


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

def get_gemini_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the Gemini API."""
    return OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def call_gemini(client: OpenAI, prompt: str, system: str = "", retries: int = SCORE_RETRIES) -> str:
    """
    Call Gemini 2.5 Flash via the OpenAI-compatible endpoint.
    Retries on transient failures.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, retries + 2):
        try:
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("Gemini call attempt %d failed: %s", attempt, exc)
            if attempt <= retries:
                time.sleep(2 ** attempt)  # Exponential back-off
            else:
                raise


# ---------------------------------------------------------------------------
# Relevance Scoring
# ---------------------------------------------------------------------------

SCORING_SYSTEM = textwrap.dedent("""\
    You are an expert researcher in 3D Computer Vision, Neural Radiance Fields (NeRF),
    and Gaussian Splatting. Your task is to score the relevance of an academic paper
    to the field of 3D Computer Vision and NeRF on a scale from 1 to 10.

    Scoring guide:
    9-10: Directly advances NeRF, Gaussian Splatting, 3D reconstruction, novel-view synthesis,
          or closely related 3D scene representation methods.
    7-8:  Strongly related — e.g., depth estimation, 3D object detection, scene understanding,
          implicit neural representations, or rendering techniques.
    5-6:  Moderately related — general computer vision with some 3D relevance.
    3-4:  Tangentially related — 2D vision tasks, image processing.
    1-2:  Unrelated or only superficially mentions the keywords.

    Respond with ONLY a single integer between 1 and 10. No explanation.
""")


def score_paper(client: OpenAI, paper: dict) -> int:
    """Score a single paper for relevance to 3D Computer Vision / NeRF."""
    prompt = (
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract']}\n\n"
        "Score (1-10):"
    )
    try:
        raw = call_gemini(client, prompt, system=SCORING_SYSTEM)
        # Extract the first integer found in the response
        for token in raw.split():
            token_clean = token.strip(".,;:")
            if token_clean.isdigit():
                score = int(token_clean)
                return max(1, min(10, score))
        log.warning("Could not parse score from response '%s'; defaulting to 5.", raw)
        return 5
    except Exception as exc:
        log.error("Scoring failed for '%s': %s", paper["title"][:60], exc)
        return 0


def score_and_filter_papers(papers: list[dict], threshold: int = RELEVANCE_THRESHOLD) -> list[dict]:
    """Score all papers and return those at or above the threshold."""
    if not papers:
        return []

    client = get_gemini_client()
    scored = []

    log.info("Scoring %d papers with Gemini %s ...", len(papers), GEMINI_MODEL)
    for i, paper in enumerate(papers, 1):
        score = score_paper(client, paper)
        paper["score"] = score
        log.info("  [%d/%d] Score %d/10 — %s", i, len(papers), score, paper["title"][:70])
        scored.append(paper)
        time.sleep(0.3)  # Gentle rate limiting

    filtered = [p for p in scored if p["score"] >= threshold]
    log.info(
        "Filtered to %d papers (score >= %d) from %d total.",
        len(filtered), threshold, len(scored),
    )
    # Sort by score descending
    filtered.sort(key=lambda p: p["score"], reverse=True)
    return filtered


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = textwrap.dedent("""\
    You are a senior researcher in 3D Computer Vision and Neural Radiance Fields.
    You write clear, insightful technical summaries for an expert audience.
    Focus on novel contributions, key trends, and practical implications.
""")


def build_summary_prompt(papers: list[dict]) -> str:
    """Build the prompt for the digest summary."""
    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += (
            f"--- Paper {i} (Score: {p['score']}/10) ---\n"
            f"Title: {p['title']}\n"
            f"Abstract: {p['abstract']}\n\n"
        )

    return (
        "Below are recent ArXiv papers in 3D Computer Vision and NeRF, "
        "ranked by relevance score.\n\n"
        f"{papers_text}"
        "Summarize the key developments, trends and breakthroughs from these papers "
        "in bullet points. Group related themes together. Be specific and technical. "
        "Highlight the most impactful contributions."
    )


def generate_summary(papers: list[dict]) -> str:
    """Generate a bullet-point digest summary using Gemini."""
    if not papers:
        return "No relevant papers found in the last 24 hours."

    client = get_gemini_client()
    prompt = build_summary_prompt(papers)
    log.info("Generating digest summary with Gemini %s ...", GEMINI_MODEL)
    summary = call_gemini(client, prompt, system=SUMMARY_SYSTEM)
    return summary


# ---------------------------------------------------------------------------
# HTML Email Formatting
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ArXiv Research Digest</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background-color: #f4f6f9;
      margin: 0; padding: 0; color: #1a1a2e;
    }}
    .wrapper {{
      max-width: 720px; margin: 32px auto; background: #ffffff;
      border-radius: 12px; overflow: hidden;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }}
    .header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      padding: 36px 40px; text-align: center;
    }}
    .header h1 {{
      color: #e94560; margin: 0 0 6px; font-size: 26px; letter-spacing: 1px;
    }}
    .header p {{
      color: #a8b2d8; margin: 0; font-size: 14px;
    }}
    .badge {{
      display: inline-block; background: #e94560; color: #fff;
      border-radius: 20px; padding: 3px 12px; font-size: 12px;
      font-weight: 600; margin-top: 10px;
    }}
    .section {{
      padding: 32px 40px;
    }}
    .section-title {{
      font-size: 18px; font-weight: 700; color: #0f3460;
      border-left: 4px solid #e94560; padding-left: 12px;
      margin: 0 0 20px;
    }}
    .summary-box {{
      background: #f8f9ff; border: 1px solid #e0e4f0;
      border-radius: 8px; padding: 24px; line-height: 1.7;
      font-size: 14px; color: #2d3561;
      white-space: pre-wrap;
    }}
    .paper-card {{
      border: 1px solid #e8ecf4; border-radius: 8px;
      padding: 20px; margin-bottom: 16px;
      transition: box-shadow 0.2s;
    }}
    .paper-card:hover {{ box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
    .paper-title {{
      font-size: 15px; font-weight: 600; color: #1a1a2e;
      margin: 0 0 8px; line-height: 1.4;
    }}
    .paper-title a {{
      color: #0f3460; text-decoration: none;
    }}
    .paper-title a:hover {{ text-decoration: underline; }}
    .paper-meta {{
      font-size: 12px; color: #7a8099; margin-bottom: 10px;
    }}
    .score-badge {{
      display: inline-block; border-radius: 4px; padding: 2px 8px;
      font-size: 11px; font-weight: 700; margin-right: 8px;
    }}
    .score-high   {{ background: #d4edda; color: #155724; }}
    .score-medium {{ background: #fff3cd; color: #856404; }}
    .score-low    {{ background: #f8d7da; color: #721c24; }}
    .abstract {{
      font-size: 13px; color: #4a5568; line-height: 1.6;
      display: -webkit-box; -webkit-line-clamp: 4;
      -webkit-box-orient: vertical; overflow: hidden;
    }}
    .stats-row {{
      display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap;
    }}
    .stat-box {{
      flex: 1; min-width: 120px; background: #f0f4ff;
      border-radius: 8px; padding: 16px; text-align: center;
    }}
    .stat-number {{ font-size: 28px; font-weight: 700; color: #e94560; }}
    .stat-label  {{ font-size: 12px; color: #7a8099; margin-top: 4px; }}
    .keywords {{
      display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px;
    }}
    .keyword-tag {{
      background: #e8f4fd; color: #0f3460; border-radius: 20px;
      padding: 4px 14px; font-size: 12px; font-weight: 500;
    }}
    .divider {{
      border: none; border-top: 1px solid #e8ecf4; margin: 0 40px;
    }}
    .footer {{
      padding: 24px 40px; text-align: center;
      font-size: 12px; color: #9aa0b4; background: #f8f9ff;
    }}
    .footer a {{ color: #0f3460; text-decoration: none; }}
  </style>
</head>
<body>
<div class="wrapper">

  <!-- Header -->
  <div class="header">
    <h1>ArXiv Research Digest</h1>
    <p>Daily intelligence on 3D Computer Vision &amp; NeRF</p>
    <span class="badge">{date_str}</span>
  </div>

  <!-- Stats -->
  <div class="section">
    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-number">{total_fetched}</div>
        <div class="stat-label">Papers Fetched</div>
      </div>
      <div class="stat-box">
        <div class="stat-number">{total_filtered}</div>
        <div class="stat-label">High-Relevance Papers</div>
      </div>
      <div class="stat-box">
        <div class="stat-number">{avg_score}</div>
        <div class="stat-label">Avg Relevance Score</div>
      </div>
    </div>

    <div class="keywords">
      {keyword_tags}
    </div>
  </div>

  <hr class="divider">

  <!-- AI Summary -->
  <div class="section">
    <h2 class="section-title">AI-Generated Digest Summary</h2>
    <div class="summary-box">{summary}</div>
  </div>

  <hr class="divider">

  <!-- Paper Cards -->
  <div class="section">
    <h2 class="section-title">Top Papers ({total_filtered} papers, score &ge; {threshold})</h2>
    {paper_cards}
  </div>

  <!-- Footer -->
  <div class="footer">
    <p>
      Generated by <strong>ArXiv Research Digest Agent</strong> &bull;
      Powered by <a href="https://deepmind.google/technologies/gemini/">Gemini 2.5 Flash</a>
    </p>
    <p>
      Architected with Claude &bull; Built with Manus AI &bull;
      Automated via GitHub Actions
    </p>
    <p style="margin-top:8px; color:#bbb;">
      Papers sourced from <a href="https://arxiv.org">arXiv.org</a> &bull;
      This digest is for research purposes only.
    </p>
  </div>

</div>
</body>
</html>
"""


def score_class(score: int) -> str:
    if score >= 8:
        return "score-high"
    if score >= 6:
        return "score-medium"
    return "score-low"


def format_authors(authors: list[str], max_authors: int = 4) -> str:
    if not authors:
        return "Unknown authors"
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + f" +{len(authors) - max_authors} more"


def build_paper_card(paper: dict) -> str:
    score = paper.get("score", 0)
    cls   = score_class(score)
    authors_str = format_authors(paper.get("authors", []))
    published   = paper.get("published", "")[:10]
    abstract    = paper.get("abstract", "")[:600] + ("..." if len(paper.get("abstract", "")) > 600 else "")
    url         = paper.get("url", "#")
    title       = paper.get("title", "Untitled")

    return f"""\
<div class="paper-card">
  <div class="paper-title">
    <a href="{url}" target="_blank">{title}</a>
  </div>
  <div class="paper-meta">
    <span class="score-badge {cls}">Score: {score}/10</span>
    {authors_str} &bull; {published}
  </div>
  <div class="abstract">{abstract}</div>
</div>"""


def format_html_email(
    papers: list[dict],
    summary: str,
    total_fetched: int,
    threshold: int = RELEVANCE_THRESHOLD,
) -> str:
    """Render the full HTML email from the template."""
    date_str  = datetime.now(timezone.utc).strftime("%B %d, %Y")
    avg_score = (
        f"{sum(p['score'] for p in papers) / len(papers):.1f}"
        if papers else "N/A"
    )
    keyword_tags = "".join(
        f'<span class="keyword-tag">{kw}</span>' for kw in KEYWORDS
    )
    paper_cards = "\n".join(build_paper_card(p) for p in papers)

    return HTML_TEMPLATE.format(
        date_str=date_str,
        total_fetched=total_fetched,
        total_filtered=len(papers),
        avg_score=avg_score,
        keyword_tags=keyword_tags,
        summary=summary,
        paper_cards=paper_cards,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Gmail SMTP Sender
# ---------------------------------------------------------------------------

def send_email(html_body: str, subject: str = None) -> None:
    """Send the HTML digest email via Gmail SMTP."""
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        log.error("Gmail credentials not set. Skipping email send.")
        return

    if subject is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subject  = f"ArXiv Research Digest — {date_str}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = GMAIL_ADDRESS

    # Plain-text fallback
    plain_text = (
        "Your ArXiv Research Digest is ready. "
        "Please view this email in an HTML-capable client."
    )
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    log.info("Sending email to %s ...", GMAIL_ADDRESS)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())
        log.info("Email sent successfully.")
    except smtplib.SMTPAuthenticationError:
        log.error(
            "Gmail authentication failed. Ensure GMAIL_APP_PASSWORD is a valid App Password "
            "(not your regular Gmail password). See: https://myaccount.google.com/apppasswords"
        )
        raise
    except Exception as exc:
        log.error("Failed to send email: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("ArXiv Research Digest Agent — %s UTC",
             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # Validate required env vars
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not GMAIL_ADDRESS:
        missing.append("GMAIL_ADDRESS")
    if not GMAIL_APP_PASSWORD:
        missing.append("GMAIL_APP_PASSWORD")
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)

    # Step 1: Fetch papers from ArXiv
    all_papers = []
    for keyword in KEYWORDS:
        papers = fetch_arxiv_papers(keyword)
        all_papers.extend(papers)

    total_fetched = len(all_papers)
    log.info("Total papers fetched (before dedup): %d", total_fetched)

    all_papers = deduplicate_papers(all_papers)
    total_unique = len(all_papers)

    if not all_papers:
        log.warning("No papers found in the last 24 hours. Sending empty digest.")
        summary = "No new papers were found in the last 24 hours for the configured keywords."
        html_body = format_html_email([], summary, total_fetched=0)
        send_email(html_body)
        return

    # Step 2: Score and filter
    filtered_papers = score_and_filter_papers(all_papers, threshold=RELEVANCE_THRESHOLD)

    if not filtered_papers:
        log.warning("No papers passed the relevance threshold of %d.", RELEVANCE_THRESHOLD)
        summary = (
            f"No papers scored {RELEVANCE_THRESHOLD}/10 or above for relevance to "
            "3D Computer Vision and NeRF today."
        )
        html_body = format_html_email([], summary, total_fetched=total_unique)
        send_email(html_body)
        return

    # Step 3: Generate summary
    summary = generate_summary(filtered_papers)

    # Step 4: Format and send
    html_body = format_html_email(
        filtered_papers, summary, total_fetched=total_unique
    )

    # Save a local copy for debugging
    output_path = "/tmp/arxiv_digest_latest.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_body)
    log.info("HTML digest saved locally to %s", output_path)

    send_email(html_body)

    log.info("=" * 60)
    log.info("Digest complete. %d papers included.", len(filtered_papers))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
