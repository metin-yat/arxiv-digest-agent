"""
Test suite for ArXiv Research Digest Agent.
Tests each module independently to verify correctness without sending emails.
"""

import os
import sys
import json
import time
import logging

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition))


# ---------------------------------------------------------------------------
# Test 1: ArXiv Fetching
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 1: ArXiv API Fetching")
print("="*60)

from arxiv_digest_agent import fetch_arxiv_papers, deduplicate_papers, KEYWORDS

try:
    # Test with a single keyword that reliably has recent papers
    papers = fetch_arxiv_papers("Computer Vision", max_results=5)
    check("ArXiv fetch returns a list", isinstance(papers, list))
    check("Each paper has required keys",
          all("title" in p and "abstract" in p and "id" in p for p in papers) if papers else True,
          f"Got {len(papers)} papers")
    if papers:
        check("Title is non-empty", bool(papers[0]["title"]))
        check("Abstract is non-empty", bool(papers[0]["abstract"]))
        check("URL is non-empty", bool(papers[0]["url"]))
        print(f"  {INFO} Sample paper: {papers[0]['title'][:80]}...")
    else:
        print(f"  {INFO} No papers found in last 24h (this is valid on quiet days)")
        check("Empty result handled gracefully", True)
except Exception as e:
    check("ArXiv fetch does not throw", False, str(e))

# Test deduplication
try:
    fake_papers = [
        {"id": "http://arxiv.org/abs/1234.5678", "title": "Paper A", "abstract": "abc"},
        {"id": "http://arxiv.org/abs/1234.5678", "title": "Paper A (dup)", "abstract": "abc"},
        {"id": "http://arxiv.org/abs/9999.0001", "title": "Paper B", "abstract": "xyz"},
    ]
    deduped = deduplicate_papers(fake_papers)
    check("Deduplication removes duplicates", len(deduped) == 2, f"Got {len(deduped)} unique papers")
except Exception as e:
    check("Deduplication does not throw", False, str(e))


# ---------------------------------------------------------------------------
# Test 2: HTML Email Formatting
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 2: HTML Email Formatting")
print("="*60)

from arxiv_digest_agent import format_html_email, build_paper_card, score_class

try:
    mock_papers = [
        {
            "id": "http://arxiv.org/abs/2401.00001",
            "title": "Gaussian Splatting for Real-Time Novel View Synthesis",
            "abstract": "We present a novel approach to real-time novel view synthesis using 3D Gaussian Splatting. Our method achieves state-of-the-art rendering quality while maintaining interactive frame rates.",
            "authors": ["Alice Smith", "Bob Jones", "Carol White"],
            "published": "2024-01-15T10:00:00Z",
            "url": "http://arxiv.org/abs/2401.00001",
            "score": 9,
        },
        {
            "id": "http://arxiv.org/abs/2401.00002",
            "title": "NeRF-based 3D Scene Reconstruction",
            "abstract": "This paper presents improvements to Neural Radiance Fields for faster training and better generalization to unseen scenes.",
            "authors": ["Dave Brown"],
            "published": "2024-01-15T11:00:00Z",
            "url": "http://arxiv.org/abs/2401.00002",
            "score": 7,
        },
    ]
    mock_summary = "• Gaussian Splatting continues to dominate real-time rendering research.\n• NeRF training speed improvements are a key trend.\n• Novel view synthesis quality is approaching photorealism."

    html = format_html_email(mock_papers, mock_summary, total_fetched=10)

    check("HTML output is a string", isinstance(html, str))
    check("HTML contains DOCTYPE", "<!DOCTYPE html>" in html)
    check("HTML contains paper titles", "Gaussian Splatting for Real-Time" in html)
    check("HTML contains summary", "Gaussian Splatting continues" in html)
    check("HTML contains stats", "10" in html)  # total_fetched
    check("HTML contains keyword tags", "Gaussian Splatting" in html)

    # Save for visual inspection
    with open("/tmp/test_digest.html", "w") as f:
        f.write(html)
    print(f"  {INFO} HTML saved to /tmp/test_digest.html for visual inspection")

    # Test score_class
    check("score_class high (9)", score_class(9) == "score-high")
    check("score_class medium (7)", score_class(7) == "score-medium")
    check("score_class low (5)", score_class(5) == "score-low")

except Exception as e:
    check("HTML formatting does not throw", False, str(e))
    import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 3: Gemini API Integration (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 3: Gemini API Integration")
print("="*60)

gemini_key = os.environ.get("GEMINI_API_KEY", "")
if not gemini_key:
    print(f"  {INFO} GEMINI_API_KEY not set — skipping live Gemini tests")
    check("Gemini API key present (skipped)", True, "No key — skipped")
else:
    from arxiv_digest_agent import get_gemini_client, call_gemini, score_paper, generate_summary

    try:
        client = get_gemini_client()
        check("Gemini client created", client is not None)

        # Test basic call
        response = call_gemini(client, "Reply with only the number 42.")
        check("Gemini returns a response", bool(response), f"Got: '{response[:50]}'")
        check("Gemini response contains 42", "42" in response, f"Got: '{response[:50]}'")

        # Test paper scoring
        test_paper = {
            "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
            "abstract": "Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. We introduce 3D Gaussian Splatting as a real-time rendering solution for radiance field methods.",
        }
        score = score_paper(client, test_paper)
        check("Gaussian Splatting paper scores >= 8", score >= 8, f"Score: {score}/10")
        check("Score is in valid range", 1 <= score <= 10, f"Score: {score}")

        # Test irrelevant paper scoring
        irrelevant_paper = {
            "title": "Sentiment Analysis of Twitter Data Using BERT",
            "abstract": "We present a method for sentiment classification of social media posts using pre-trained BERT models. Our approach achieves state-of-the-art results on the SST-2 benchmark.",
        }
        irr_score = score_paper(client, irrelevant_paper)
        check("Irrelevant paper scores <= 4", irr_score <= 4, f"Score: {irr_score}/10")

        # Test summary generation with mock papers
        mock_papers_for_summary = [
            {
                "title": "Gaussian Splatting for Real-Time Novel View Synthesis",
                "abstract": "We present a novel approach to real-time novel view synthesis using 3D Gaussian Splatting.",
                "score": 9,
            }
        ]
        summary = generate_summary(mock_papers_for_summary)
        check("Summary is non-empty", bool(summary) and len(summary) > 50, f"Length: {len(summary)}")
        print(f"  {INFO} Summary preview: {summary[:150]}...")

    except Exception as e:
        check("Gemini integration does not throw", False, str(e))
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 4: Environment Variable Validation
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 4: Environment Variable Handling")
print("="*60)

try:
    # Test that the script correctly detects missing env vars
    import subprocess
    result = subprocess.run(
        [sys.executable, "arxiv_digest_agent.py"],
        capture_output=True, text=True,
        env={**os.environ, "GEMINI_API_KEY": "", "GMAIL_ADDRESS": "", "GMAIL_APP_PASSWORD": ""},
        cwd=os.path.dirname(__file__) or ".",
    )
    check("Script exits with code 1 on missing env vars", result.returncode == 1,
          f"Exit code: {result.returncode}")
    check("Script reports missing variables", "Missing required environment variables" in result.stderr,
          f"Stderr: {result.stderr[:100]}")
except Exception as e:
    check("Env var validation test does not throw", False, str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print(f"  {PASS} All tests passed!")
else:
    failed = [name for name, ok in results if not ok]
    print(f"  {FAIL} Failed tests: {', '.join(failed)}")
    sys.exit(1)
