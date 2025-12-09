"""Combined agent that analyzes page and decides action in a single LLM call."""

import json
import re
import textwrap
from typing import Any, Dict, List


def model_supports_vision_json(model: str) -> bool:
    return "4o" in (model or "") or "5" in (model or "")


def parse_llm_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", content or "")
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_text(text: str, limit: int = 3000) -> str:
    if not text:
        return ""
    return text[:limit]


async def analyze_and_act(
        client,
        model: str,
        goal: str,
        digest: List[Dict[str, Any]],
        state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combined function that analyzes the page AND decides the next action in ONE LLM call.
    This is ~2x faster than separate extract_meaning + choose_action calls.
    """
    system = textwrap.dedent(
        """
        You are a human-like browsing agent. Analyze the page and decide ONE immediate action.
        
        RULES:
        - If a cookie/consent BUTTON (tag=button, not link) appears, dismiss it first.
        - Do NOT click cookie/consent links (tag=a) - those are informational pages.
        - Use hover_click for dropdown nav menus.
        - Use "type" to fill input fields (forms, search boxes).
        - Use "select_option" to select from dropdowns (select tags).
        - Use "solve_image_captcha" for distorted text captchas.
        - Prefer nav refs that match the goal keywords.
        - REMOVAL CHECK: If you clicked "Remove" and the button changed to "Add to cart", the item is removed. Goal complete.
        
        Return JSON only:
        {
          "action": "click" | "hover_click" | "type" | "select_option" | "solve_image_captcha" | "done",
          "primary_ref": <number or null>,
          "secondary_ref": <number or null>,
          "value": "<text to type/select if action=type/select_option>",
          "notes": "<brief reasoning>"
        }
        """
    ).strip()

    # Build compact digest string
    digest_str = clamp_text(json.dumps(digest, indent=2), 3000)

    user = f"""Goal: {goal}

Current URL: {state.get('pageUrl', 'unknown')}
Page Title: {state.get('pageTitle', 'unknown')}

Available elements (ref, tag, text):
{digest_str}

Decide the single best action to progress toward the goal."""

    include_image = model_supports_vision_json(model) and state.get("screenshotBase64")
    user_content = [{"type": "text", "text": user}]
    if include_image:
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state.get('screenshotBase64', '')}"}}
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    kwargs = {"model": model, "messages": messages}
    if model_supports_vision_json(model):
        kwargs["response_format"] = {"type": "json_object"}

    # BULLETPROOF: Retry logic for LLM API calls
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            return parse_llm_json(content)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                import asyncio
                wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                print(f"  LLM API retry {attempt + 1}/{max_retries} after {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  LLM API failed after {max_retries} attempts: {last_error}")
                # Return a safe fallback action
                return {"action": "done", "notes": f"LLM API error: {str(last_error)[:50]}"}
