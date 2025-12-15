import asyncio
import json
import textwrap
from typing import Tuple, Dict, Any


class VerifierAgent:
    """
    Agent responsible for judging if the acceptance criteria were met based on the final page state.
    """

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    async def verify(
            self,
            story: Dict[str, Any],
            final_url: str,
            visible_text: str
    ) -> Tuple[bool, str]:
        """
        Evaluate if the story criteria are met.
        Returns (success: bool, reason: str).
        """
        criteria = story.get("acceptance_criteria") or []
        description = story.get("description", "")
        title = story.get("title", "")
        test_data = story.get("test_data", {})
        expected_failure = test_data.get("expected_failure", False)
        expected_url = test_data.get("expected_url", "")
        expected_text = test_data.get("expected_text", "")

        # 1. Heuristic fallback for strict URL/text expectations
        # This is a fast-path check usually defined in strict unit tests
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

        # 2. LLM-based evaluation with retry logic
        system_prompt = textwrap.dedent(
            """
            You are a QA Checker Agent.
            Your job is to decide if a User Story is satisfied based on the Final Page State.
            
            INPUT:
            - User Story (Title, Criteria)
            - Final URL
            - Final Page Text (Truncated)
            - Test Data (Expected Failure?)
            
            LOGIC:
            - If expected_failure=True (Negative Test), you PASS only if the valid error/failure state is reached.
            - If normal test, you PASS only if all acceptance criteria are met.
            
            OUTPUT:
            - Return JSON ONLY: {"status": "PASS" | "FAIL", "reason": "concise explanation"}
            """
        ).strip()

        user_prompt = (
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Acceptance Criteria:\n- " + "\n- ".join(criteria) + "\n"
                                                                      f"Test Data: {json.dumps(test_data)}\n\n"
                                                                      f"Final URL: {final_url}\n"
                                                                      f"Final Visible Text (First 2000 chars):\n{visible_text[:2000]}\n"
        )

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                content = resp.choices[0].message.content or ""

                # Parse JSON
                try:
                    parsed = json.loads(content)
                    status = parsed.get("status", "").upper()
                    reason = parsed.get("reason", "")
                    return status == "PASS", reason or "LLM decision"
                except json.JSONDecodeError:
                    # Fallback for simple string response
                    normalized = content.strip().lower()
                    passed = "pass" in normalized and "fail" not in normalized
                    return passed, content[:200]

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 2)

        # 3. Final Fallback if LLM fails completely
        return self._heuristic_fallback(criteria, final_url, visible_text, str(last_error))

    def _heuristic_fallback(self, criteria: list, final_url: str, visible_text: str, error_msg: str) -> Tuple[
        bool, str]:
        """
        Simple keyword matching if LLM is down.
        """
        criteria_text = " ".join(criteria).lower()
        success = True

        if "client" in criteria_text:
            success = "client" in final_url
        elif "work" in criteria_text or "portfolio" in criteria_text:
            success = "work" in final_url
        elif "insight" in criteria_text or "blog" in criteria_text:
            success = "insight" in final_url
        elif "contact" in criteria_text or "form" in criteria_text:
            success = "email" in visible_text or "enquiry" in visible_text or "send" in visible_text

        return success, f"Fallback heuristic (LLM error: {error_msg})"
