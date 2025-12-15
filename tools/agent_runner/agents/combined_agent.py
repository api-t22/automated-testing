import asyncio
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


class NavigatorAgent:
    """
    The Core Navigator Agent.
    Decides the next action (Click, Type, Goto) based on Goal + State.
    Uses 'analyze_and_act' logic encapsulated in a class.
    """

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    async def analyze_and_act(
            self,
            goal: str,
            digest: List[Dict[str, Any]],
            state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the page and decide action in ONE LLM call.
        """
        system = textwrap.dedent(
            """
            You are a human-like browsing agent. Analyze the page and decide ONE immediate action.
            
            RULES:
            - If a cookie/consent BUTTON (tag=button, not link) appears, dismiss it first.
            - Do NOT click cookie/consent links (tag=a) - those are informational pages.
            - Use hover_click for dropdown nav menus.
            - Use "type" to fill input fields (forms, search boxes).
            - Use "goto" to navigate directly to a specific URL if the goal explicitly mentions a full URL (e.g. "Navigate to /inventory.html").
            - Use "select_option" to select from dropdowns (select tags).
            - Use "solve_image_captcha" for distorted text captchas.
            - Prefer nav refs that match the goal keywords.
            - DYNAMIC JUDGE:
              - You are the JUDGE of this test. You must decide if the goal is completed.
              - "success": The goal is fundamentally achieved (e.g. Order Placed, Item Removed, User Logged In).
              - "failed": The goal is impossible or a critical error occurred (e.g. Site Crash 500, specific feature missing).
              - "running": The goal is not yet complete.
              - NEGATIVE TESTS: If the goal expects failure/error and you see an error message, status is "success" (Test Passed).
            - AUTONOMOUS TESTER PRINCIPLES (THINK LIKE A QA ENGINEER):
              - PRINCIPLE 1: ESTABLISH PRECONDITIONS (CLEAN SLATE).
                - Before starting a task, ask: "Am I in the correct state?"
                - Example: If the test is "Login with invalid user", you MUST be on the Login Page. If you are already logged in, you MUST Logout first.
                - Example: If an error message is already visible from a PREVIOUS test, REFRESH the page to clear it before typing.
                - Example: If the test is "Checkout", you MUST have items in the cart and be in the Checkout flow. If not, Navigate there.
              - PRINCIPLE 2: VERIFY EFFECTS, NOT JUST ACTIONS.
                - Don't just "Click Remove". Verify the item is actually gone.
                - Don't just "Click Login". Verify you are actually on the Dashboard.
                - Use your Common Sense: If the UI updates (badges, text, icons), use that as proof.
              - PRINCIPLE 3: FORM VALIDATION.
                - When testing field validation (e.g. "First Name required"), you MUST Trigger the validation.
                - If typing doesn't remove an error, click "Continue" or Click outside the field (Blur) to force re-validation.
            - STEP TRACKING:
              - The Goal contains a list of steps/criteria.
              - Check "RECENT ACTIONS" to see what you have already done.
              - Execute the NEXT logical step strategies.
              - Do NOT repeat steps you have already completed (especially toggles).
              - CRITICAL EXECUTION RULE: You MUST COMPLETE ALL STEPS in the goal.
              - Do NOT declare "success" if you have NOT clicked the submit/continue button in a form flow.
              - Validation errors ONLY appear AFTER clicking Submit.
            - PRESERVE STATE (NO ACCIDENTAL REFRESH):
              - Do NOT use "goto" to the CURRENT page URL. This reloads the page and Wipes Form Data.
              - If you have partially typed a form (see Recent Actions), do NOT Navigate/Refresh unless explicitly stuck.
              - Continue filling the remaining fields or click Submit.
            
            Return JSON only:
            {
              "action": "click" | "hover_click" | "type" | "select_option" | "goto" | "solve_image_captcha" | "done",
              "status": "running" | "success" | "failed",
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

        include_image = model_supports_vision_json(self.model) and state.get("screenshotBase64")
        user_content = [{"type": "text", "text": user}]
        if include_image:
            user_content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{state.get('screenshotBase64', '')}"}}
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        kwargs = {"model": self.model, "messages": messages}
        if model_supports_vision_json(self.model):
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = 3

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return parse_llm_json(content)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    print(f"  LLM API retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  LLM API failed after {max_retries} attempts: {last_error}")
                    # Return a safe fallback action
                    return {"action": "done", "notes": f"LLM API error: {str(last_error)[:50]}"}
