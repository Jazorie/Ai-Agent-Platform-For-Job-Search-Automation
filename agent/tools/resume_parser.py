"""
Tool: parse_resume
Extracts structured data from raw resume text.
"""

import json
import anthropic


def parse_resume(resume_text: str) -> dict:
    client = anthropic.Anthropic()

    prompt = f"""Extract structured information from this resume.

Resume:
{resume_text}

Respond ONLY with a JSON object (no markdown, no preamble):
{{
  "name": "<full name or null>",
  "skills": {{
    "languages": ["Python", "JavaScript", ...],
    "frameworks": ["React", "FastAPI", ...],
    "tools": ["Docker", "Git", ...],
    "other": [...]
  }},
  "experience": [
    {{
      "company": "<name>",
      "role": "<title>",
      "duration": "<e.g. Jun 2023 – Present>",
      "highlights": ["<bullet 1>", "<bullet 2>"]
    }}
  ],
  "education": [
    {{
      "institution": "<name>",
      "degree": "<e.g. B.S. Computer Science>",
      "graduation": "<year or expected>"
    }}
  ],
  "years_of_experience": <number or null>
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw, "parse_error": True}
