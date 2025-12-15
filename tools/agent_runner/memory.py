from datetime import datetime
from typing import List, Dict, Any


class SemanticMemory:
    """
    Stores the history of the Agent's journey to allow for 'Smart' querying.
    Acts as a lightweight Vector DB surrogate.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add_state(self, step: int, url: str, action: str, entities: List[str] = None, screenshot_path: str = None):
        """
        Log a state transition.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "url": url,
            "action": action,
            "entities": entities or [],
            "screenshot": screenshot_path
        }
        self.history.append(entry)
        # print(f"  [Memory] Recorded step {step} at {url}")

    def get_recent_history(self, limit: int = 5) -> str:
        """
        Returns a formatted string of recent actions for the Prompt.
        """
        if not self.history:
            return "No history yet."

        recent = self.history[-limit:]
        text = []
        for h in recent:
            text.append(f"Step {h['step']}: {h['action']} (at {h['url']})")
        return "\n".join(text)

    def query_history(self, query: str) -> str:
        """
        Simple keyword search implementation (Placeholder for Vector Search).
        """
        # TODO: Implement actual semantic search if needed.
        # For now, return relevant lines containing keywords.
        keywords = query.replace("?", "").split()
        matches = []
        for h in self.history:
            content = f"{h['action']} {h['url']} {' '.join(h['entities'])}"
            if any(k.lower() in content.lower() for k in keywords):
                matches.append(f"Step {h['step']}: {h['action']}")

        if not matches:
            return "No relevant memory found."
        return "\n".join(matches)

    def add_location(self, name: str, url: str):
        """
        Record a known location for smart navigation (e.g. 'login_page' -> '/').
        """
        # Simple tracking, could be expanded
        self.history.append(
            {"action": f"Note Location: {name}", "url": url, "timestamp": datetime.now().isoformat(), "step": 0,
             "entities": []})
