"""
Tool: score_fit
Scores how well a job description matches a resume using the Anthropic API.
Returns a structured dict with score, matched_skills, gaps, and summary.
"""

import json
import anthropic


def score_fit(job_description: str, resume_text: str) -> dict:
    client = anthropic.Anthropic()

    prompt = f"""You are a technical recruiter scoring how well a candidate matches a job.

Job Description:
{job_description}

Resume:
{resume_text}

Analyze the match and respond ONLY with a JSON object (no markdown, no preamble):
{{
  "fit_score": <integer 0-100>,
  "matched_skills": ["skill1", "skill2", ...],
  "skill_gaps": ["missing1", "missing2", ...],
  "summary": "<2-3 sentence summary of fit>",
  "recommendation": "strong_match" | "good_match" | "partial_match" | "poor_match"
}}

Scoring guide:
- 85-100: Candidate meets or exceeds nearly all requirements
- 70-84: Strong match with minor gaps
- 50-69: Partial match, notable gaps but transferable skills
- Below 50: Significant misalignment"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Fast + cheap for tool calls
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "fit_score": 0,
            "matched_skills": [],
            "skill_gaps": [],
            "summary": "Failed to parse fit score response.",
            "recommendation": "poor_match",
            "raw": raw,
        }
