"""
Live pipeline test using the sandbox's OpenAI-compatible API (Gemini 2.5 Flash).
This validates the full fetch → score → filter → summarize → HTML pipeline
without actually sending an email.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Use the sandbox's OPENAI_API_KEY but override the base_url to use Gemini
# We patch the module-level variables before importing the functions
import arxiv_digest_agent as agent

# Override to use the sandbox key and Gemini endpoint
SANDBOX_KEY = os.environ.get("OPENAI_API_KEY", "")
if not SANDBOX_KEY:
    print("OPENAI_API_KEY not available in sandbox — cannot run live test")
    sys.exit(0)

# Monkey-patch the agent to use the sandbox key with Gemini endpoint
agent.GEMINI_API_KEY = SANDBOX_KEY

from openai import OpenAI

def get_test_client():
    """Return client using sandbox key pointed at Gemini."""
    return OpenAI(
        api_key=SANDBOX_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

# Override the client factory
agent.get_gemini_client = get_test_client

print("\n" + "="*60)
print("LIVE PIPELINE TEST (Gemini 2.5 Flash)")
print("="*60)

# Step 1: Fetch real papers
print("\n[Step 1] Fetching ArXiv papers...")
all_papers = []
for keyword in agent.KEYWORDS:
    papers = agent.fetch_arxiv_papers(keyword, max_results=10)
    all_papers.extend(papers)

all_papers = agent.deduplicate_papers(all_papers)
print(f"  Total unique papers: {len(all_papers)}")

if not all_papers:
    print("  No papers found in last 24h — using mock papers for testing")
    all_papers = [
        {
            "id": "http://arxiv.org/abs/2401.00001",
            "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
            "abstract": "Radiance Field methods have recently revolutionized novel-view synthesis. We introduce 3D Gaussian Splatting as a real-time rendering solution that achieves state-of-the-art visual quality while maintaining interactive frame rates on a GPU.",
            "authors": ["Bernhard Kerbl", "Georgios Kopanas"],
            "published": "2024-01-15T10:00:00Z",
            "url": "http://arxiv.org/abs/2401.00001",
        },
        {
            "id": "http://arxiv.org/abs/2401.00002",
            "title": "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections",
            "abstract": "We present a learning-based method for synthesizing novel views of complex scenes using only unstructured collections of in-the-wild photographs.",
            "authors": ["Ricardo Martin-Brualla"],
            "published": "2024-01-15T11:00:00Z",
            "url": "http://arxiv.org/abs/2401.00002",
        },
        {
            "id": "http://arxiv.org/abs/2401.00003",
            "title": "Sentiment Analysis of Social Media Posts",
            "abstract": "We propose a BERT-based approach for classifying sentiment in Twitter data. Our method achieves 94% accuracy on the SST-2 benchmark.",
            "authors": ["John Doe"],
            "published": "2024-01-15T12:00:00Z",
            "url": "http://arxiv.org/abs/2401.00003",
        },
    ]

# Step 2: Score and filter (limit to 5 papers for speed)
test_papers = all_papers[:5]
print(f"\n[Step 2] Scoring {len(test_papers)} papers with Gemini 2.5 Flash...")
filtered = agent.score_and_filter_papers(test_papers, threshold=agent.RELEVANCE_THRESHOLD)
print(f"  Papers passing threshold ({agent.RELEVANCE_THRESHOLD}+): {len(filtered)}")

for p in filtered:
    print(f"    Score {p['score']}/10 — {p['title'][:70]}")

# Step 3: Generate summary
print(f"\n[Step 3] Generating digest summary...")
if filtered:
    summary = agent.generate_summary(filtered)
    print(f"  Summary ({len(summary)} chars):")
    print("  " + "\n  ".join(summary[:500].split("\n")))
    if len(summary) > 500:
        print("  ...")
else:
    summary = "No relevant papers found."
    print(f"  {summary}")

# Step 4: Format HTML
print(f"\n[Step 4] Formatting HTML email...")
html = agent.format_html_email(filtered, summary, total_fetched=len(all_papers))
print(f"  HTML size: {len(html):,} bytes")

# Save for inspection
output_path = "/tmp/arxiv_digest_live_test.html"
with open(output_path, "w") as f:
    f.write(html)
print(f"  Saved to: {output_path}")

# Validate HTML structure
checks = [
    ("Contains DOCTYPE",       "<!DOCTYPE html>" in html),
    ("Contains paper cards",   "paper-card" in html),
    ("Contains summary",       len(summary) > 10 and summary[:20] in html),
    ("Contains header",        "ArXiv Research Digest" in html),
    ("Contains footer",        "Gemini 2.5 Flash" in html),
]

print(f"\n[Step 5] HTML Validation:")
all_ok = True
for name, ok in checks:
    status = "\033[92m[PASS]\033[0m" if ok else "\033[91m[FAIL]\033[0m"
    print(f"  {status} {name}")
    if not ok:
        all_ok = False

print("\n" + "="*60)
if all_ok:
    print("\033[92mLIVE PIPELINE TEST PASSED\033[0m")
else:
    print("\033[91mLIVE PIPELINE TEST FAILED\033[0m")
    sys.exit(1)
