import asyncio
import json
import textwrap
from typing import List, Any


class SummariserAgent:
    """
    Agent responsible for synthesizing raw investigation findings into a cohesive narrative.
    Uses a standard design pattern compatible with ObserverAgent and CombinedAgent.
    """

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    async def summarize_investigation(self, findings: List[str], error_msg: str) -> str:
        """
        Summarize a list of investigation findings into a cohesive narrative.
        Includes retry logic for robustness.
        """
        system = textwrap.dedent(
            """
            You are a Lead QA Engineer.
            Your task is to synthesize raw investigation logs into a SINGLE, CONCISE narrative sentence.
            Focus on the ROOT CAUSE or the STATE CONTRADICTION found.
            Do NOT list steps individually.
            Make it read like a human diagnosis.
            """
        ).strip()

        user_prompt = (
            f"Original Failure: {error_msg}\n\n"
            f"Investigation Findings:\n{json.dumps(findings, indent=2)}\n\n"
            f"Write the one-sentence summary:"
        )

        # Standard Agent Retry Loop
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Sync client call (run in executor if needed, but here simple sync is fine)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)
                else:
                    # Graceful Fallback
                    return f"{'; '.join(findings)} (Summary Error: {str(last_error)})"

        return "; ".join(findings)
