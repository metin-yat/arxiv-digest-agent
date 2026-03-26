# ArXiv Research Digest Agent

An autonomous Python agent that curates daily research updates from ArXiv, scores them for relevance using Google's Gemini 2.5 Flash, and delivers a beautifully formatted HTML digest directly to your Gmail inbox.

## The Problem

Keeping up with the firehose of daily academic publications on ArXiv is overwhelming, especially in fast-moving fields like 3D Computer Vision, Neural Radiance Fields (NeRF), and Gaussian Splatting. Researchers often spend hours manually sifting through titles and abstracts to find the few papers that actually matter to their specific niche. Keyword alerts are too noisy, and manual filtering is too slow.

## The Solution

This agent automates the entire literature review pipeline:
1. **Fetch**: Pulls all papers published in the last 24 hours matching specific keywords ("Gaussian Splatting", "NeRF", "Computer Vision").
2. **Score**: Uses Gemini 2.5 Flash to read each abstract and score its relevance (1-10) specifically to 3D Computer Vision and NeRF.
3. **Filter**: Discards papers scoring below a configurable threshold (default: 6).
4. **Summarize**: Prompts Gemini to generate a concise, bulleted summary of the key trends and breakthroughs from the top papers.
5. **Deliver**: Formats the results into a clean, responsive HTML email and sends it via Gmail SMTP.

## Tech Stack

* **Architected with Claude** (prompt engineering + system design)
* **Built with Manus AI** (agent-based development)
* **Powered by Gemini 2.5 Flash** (LLM inference)
* **Automated via GitHub Actions** (CI/CD)

## Setup Instructions

### 1. Prerequisites
* Python 3.11+
* A Google account (for Gmail SMTP)
* A Google AI Studio account (for Gemini API)

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/arxiv-digest-agent.git
cd arxiv-digest-agent
pip install -r requirements.txt
```

### 3. Configuration
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
* `GEMINI_API_KEY`: Get this from [Google AI Studio](https://aistudio.google.com/app/apikey).
* `GMAIL_ADDRESS`: Your full Gmail address (e.g., `you@gmail.com`).
* `GMAIL_APP_PASSWORD`: You must generate an **App Password** for this script. Do not use your regular Gmail password.
  1. Enable 2-Step Verification on your Google account.
  2. Go to [App Passwords](https://myaccount.google.com/apppasswords).
  3. Create a new app password (e.g., name it "ArXiv Agent") and paste the 16-character code into your `.env` file.

### 4. Local Testing
Run the script manually to test the pipeline:
```bash
python arxiv_digest_agent.py
```
Check your inbox for the digest! A local copy of the HTML will also be saved to `/tmp/arxiv_digest_latest.html` for debugging.

## Automation with GitHub Actions

This repository includes a GitHub Actions workflow (`.github/workflows/daily.yml`) that runs the agent automatically every day at 08:00 UTC.

To enable this:
1. Push this code to a GitHub repository.
2. Go to your repository **Settings** > **Secrets and variables** > **Actions**.
3. Add the following **Repository secrets**:
   * `GEMINI_API_KEY`
   * `GMAIL_ADDRESS`
   * `GMAIL_APP_PASSWORD`
4. The workflow will now run automatically on schedule. You can also trigger it manually from the **Actions** tab.

## Customization

You can easily adapt this agent for other research fields by modifying the constants at the top of `arxiv_digest_agent.py`:
* `KEYWORDS`: Change the ArXiv search terms.
* `SCORING_SYSTEM`: Update the prompt to score relevance based on your specific domain.
* `SUMMARY_SYSTEM`: Adjust the persona and focus of the AI-generated summary.
* `RELEVANCE_THRESHOLD`: Raise or lower the strictness of the filter.

## Example Output
![Email Output 1](screenshots/image1.png)
![Email Output 2](screenshots/image2.png)

## License

MIT License. See `LICENSE` for details.
