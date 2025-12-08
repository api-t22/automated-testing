import json
import re
import textwrap
from typing import Any, Dict, List


def model_supports_vision_json(model: str) -> bool:
    return "4o" in (model or "")


def parse_llm_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", content or "")
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_text(text: str, limit: int = 2000) -> str:
    if not text:
        return ""
    return text[:limit]


async def choose_action(client, model: str, goal: str, digest: List[Dict[str, Any]], meaning: Dict[str, Any]) -> Dict[
    str, Any]:
    system = textwrap.dedent(
        """
        You are a human-like browsing agent. Plan one immediate action based on the meaning summary.
        If a cookie/consent BUTTON (tag=button, not link) appears in the digest, dismiss it first.
        Do NOT click cookie/consent links (tags=a) - those are informational, not dismissals.
        Use hover before click for dropdown nav.
        Use "type" action to fill input fields with text (forms, search boxes, etc.).
        Use "solve_captcha" for simple math (e.g., "3+5=?") or text captchas visible on the page.
        Use "solve_image_captcha" for image-based captchas with distorted letters (uses GPT-4 Vision).
        Prefer nav/subnav refs that match the goal (e.g., About/Clients) before generic links.
        Return JSON only:
        {
          "action": "click" | "hover_click" | "type" | "solve_captcha" | "solve_image_captcha" | "assert_text" | "assert_url" | "done",
          "primary_ref": <number or null>,
          "secondary_ref": <number or null>,
          "value": "<text to type, captcha question, or assert substring>",
          "notes": "<reasoning>"
        }
        """
    ).strip()

    user = f"""Goal: {goal}
Meaning summary (truncated): {clamp_text(json.dumps(meaning, indent=2), 2000)}
Digest (refs only): {clamp_text(json.dumps(digest, indent=2), 2000)}
"""

    include_image = model_supports_vision_json(model) and meaning.get("screenshotBase64")
    user_content = [{"type": "text", "text": user}]
    if include_image:
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{meaning.get('screenshotBase64', '')}"}}
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    kwargs = {"model": model, "messages": messages}
    if model_supports_vision_json(model):
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    return parse_llm_json(content)
