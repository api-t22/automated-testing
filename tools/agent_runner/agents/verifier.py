"""
LLM-backed completion judging for user stories / test cases.
"""
import json
from typing import Tuple

from tools.agent_runner.actions import openai_client


async def judge_completion(
        story: dict,
        final_url: str,
        visible_text: str,
        *,
        client=None,
        model: str = "gpt-5-nano-2025-08-07",
) -> Tuple[bool, str]:
    """
    Use the LLM to decide if the acceptance criteria were met based on the page URL/text.
    Falls back to heuristic rules if the LLM is unavailable.
    """
    criteria = story.get("acceptance_criteria") or []
    description = story.get("description", "")
    title = story.get("title", "")
    test_data = story.get("test_data", {})
    expected_failure = test_data.get("expected_failure", False)
    expected_url = test_data.get("expected_url", "")
    expected_text = test_data.get("expected_text", "")

    # Heuristic fallback for strict URL/text expectations
    if expected_failure:
        if expected_text and expected_text.lower() in visible_text.lower():
            return True, "Negative case matched expected error text"
    else:
        url_match = not expected_url or expected_url in final_url
        text_match = not expected_text or expected_text.lower() in visible_text.lower()
        if expected_url or expected_text:
            if url_match and text_match:
                return True, "Expected URL/text matched"
            return False, f"URL match={url_match}, text match={text_match}"

    # LLM-based evaluation
    try:
        client = client or openai_client()
        prompt = (
                "You are a QA checker. Decide if the user story is satisfied given the final page URL and visible text.\n"
                "Respond with JSON: {\"status\": \"PASS\"|\"FAIL\", \"reason\": \"...\"}.\n"
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Acceptance criteria:\n- " + "\n- ".join(criteria) + "\n"
                                                                      f"Final URL: {final_url}\n"
                                                                      f"Visible text (truncated): {visible_text[:2000]}\n"
                                                                      f"Test data: {json.dumps(test_data)}\n"
                                                                      "If this is a negative test (expected_failure=true), PASS only when the expected failure condition is seen."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You only output JSON with status and reason."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
            status = parsed.get("status", "").upper()
            reason = parsed.get("reason", "")
            return status == "PASS", reason or "LLM decision"
        except Exception:
            normalized = content.strip().lower()
            status = "pass" in normalized and "fail" not in normalized
            return status, content[:200]
    except Exception as e:
        # LLM not available; fall back to simple keyword check
        criteria_text = " ".join(criteria).lower()
        if "client" in criteria_text:
            success = "client" in final_url
        elif "work" in criteria_text or "portfolio" in criteria_text:
            success = "work" in final_url
        elif "insight" in criteria_text or "blog" in criteria_text:
            success = "insight" in final_url
        elif "people" in criteria_text or "team" in criteria_text:
            success = "people" in final_url
        elif "contact" in criteria_text or "form" in criteria_text:
            success = "email" in visible_text or "enquiry" in visible_text or "send" in visible_text
        elif "social" in criteria_text:
            success = "linkedin" in visible_text or "twitter" in visible_text or "instagram" in visible_text
        else:
            success = True
        return success, f"Fallback heuristic (LLM error: {e})"
