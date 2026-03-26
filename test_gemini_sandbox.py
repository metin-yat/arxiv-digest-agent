"""
Test Gemini 2.5 Flash via the sandbox's pre-configured OpenAI client.
The sandbox pre-configures OPENAI_API_KEY + base_url for Gemini access.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

from openai import OpenAI

# Use the sandbox's pre-configured client (key + base_url from env)
client = OpenAI()

print("\n" + "="*60)
print("GEMINI 2.5 FLASH — Sandbox API Test")
print("="*60)

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition))

# Test 1: Basic call
try:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Reply with only the number 42."}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()
    check("Gemini 2.5 Flash responds", bool(answer), f"Got: '{answer}'")
    check("Response contains 42", "42" in answer, f"Got: '{answer}'")
except Exception as e:
    check("Gemini 2.5 Flash basic call", False, str(e))

# Test 2: Score a NeRF paper
try:
    nerf_paper = """Title: 3D Gaussian Splatting for Real-Time Radiance Field Rendering

Abstract: Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce 3D Gaussian Splatting as a real-time rendering solution for radiance field methods."""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are an expert in 3D Computer Vision and NeRF. Score the relevance of this paper to 3D Computer Vision and NeRF on a scale 1-10. Reply with ONLY a single integer."},
            {"role": "user", "content": nerf_paper + "\n\nScore (1-10):"},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    score = None
    for token in raw.split():
        if token.strip(".,;:").isdigit():
            score = int(token.strip(".,;:"))
            break
    check("NeRF paper scoring works", score is not None, f"Raw: '{raw}'")
    if score is not None:
        check("NeRF paper scores >= 8", score >= 8, f"Score: {score}/10")
except Exception as e:
    check("NeRF paper scoring", False, str(e))

# Test 3: Score an irrelevant paper
try:
    irr_paper = """Title: Sentiment Analysis of Twitter Data Using BERT

Abstract: We present a method for sentiment classification of social media posts using pre-trained BERT models fine-tuned on Twitter data. Our approach achieves 94% accuracy on the SST-2 benchmark and demonstrates strong generalization across domains."""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are an expert in 3D Computer Vision and NeRF. Score the relevance of this paper to 3D Computer Vision and NeRF on a scale 1-10. Reply with ONLY a single integer."},
            {"role": "user", "content": irr_paper + "\n\nScore (1-10):"},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    score = None
    for token in raw.split():
        if token.strip(".,;:").isdigit():
            score = int(token.strip(".,;:"))
            break
    check("Irrelevant paper scoring works", score is not None, f"Raw: '{raw}'")
    if score is not None:
        check("Irrelevant paper scores <= 3", score <= 3, f"Score: {score}/10")
except Exception as e:
    check("Irrelevant paper scoring", False, str(e))

# Test 4: Summary generation
try:
    papers_text = """Paper 1 (Score: 9/10):
Title: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
Abstract: We introduce 3D Gaussian Splatting as a real-time rendering solution for radiance field methods, achieving state-of-the-art quality at interactive frame rates.

Paper 2 (Score: 8/10):
Title: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
Abstract: We present a versatile new primitive for neural graphics applications, including NeRF, signed distance functions, and neural radiance caching."""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a senior researcher in 3D Computer Vision. Write concise technical summaries."},
            {"role": "user", "content": f"Below are recent ArXiv papers:\n\n{papers_text}\n\nSummarize the key developments, trends and breakthroughs from these papers in bullet points."},
        ],
        temperature=0.2,
    )
    summary = response.choices[0].message.content.strip()
    check("Summary generation works", bool(summary) and len(summary) > 50, f"Length: {len(summary)}")
    check("Summary contains bullet points", "•" in summary or "-" in summary or "*" in summary, "Has bullets")
    print(f"\n  Summary preview:\n  {summary[:300]}...")
except Exception as e:
    check("Summary generation", False, str(e))

# Final results
print("\n" + "="*60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print(f"{PASS} All Gemini API tests passed!")
else:
    failed = [name for name, ok in results if not ok]
    print(f"{FAIL} Failed: {', '.join(failed)}")
    sys.exit(1)
