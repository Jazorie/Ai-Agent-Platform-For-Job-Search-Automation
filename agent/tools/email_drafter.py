"""
Tool: draft_email
Drafts a personalized cold outreach email for a job application.
"""

import anthropic


def draft_email(
    job_description: str,
    candidate_name: str,
    fit_summary: str,
    tone: str = "professional",
) -> dict:
    client = anthropic.Anthropic()

    tone_guide = {
        "professional": "formal, polished, confident — like a senior engineer writing to a peer",
        "conversational": "warm and direct, like reaching out to someone you met at a meetup",
        "enthusiastic": "energetic and genuine, showing real excitement about the company mission",
    }.get(tone, "professional")

    prompt = f"""Draft a cold outreach email for a job application.

Job Description:
{job_description}

Candidate Name: {candidate_name}
Fit Analysis: {fit_summary}
Tone: {tone_guide}

Rules:
- Subject line should be specific, not generic ("Following up on SWE role" is bad)
- Body should be 3-4 short paragraphs max
- Reference 1-2 specific things from the job description
- End with a clear, low-friction call to action
- Do NOT use filler phrases like "I hope this email finds you well"
- Do NOT attach a resume in the email copy — just mention it's available

Respond ONLY with a JSON object (no markdown):
{{
  "subject": "<email subject line>",
  "body": "<full email body with \\n for line breaks>",
  "word_count": <integer>
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        import json
        return json.loads(raw)
    except Exception:
        return {
            "subject": "Application — Software Engineer",
            "body": raw,
            "word_count": len(raw.split()),
        }
