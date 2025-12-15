import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from playwright.async_api import async_playwright

from app.config import Settings
from tools.agent_runner.actions import (
    apply_action,
    auto_dismiss_popups,
    capture_state,
    digest_for_action,
    load_env,
    openai_client,
)
from tools.agent_runner.agents.combined_agent import NavigatorAgent
from tools.agent_runner.agents.observer_agent import ObserverAgent
from tools.agent_runner.agents.summariser import SummariserAgent
from tools.agent_runner.agents.verifier import VerifierAgent
from tools.agent_runner.memory import SemanticMemory
from tools.agent_runner.page_model import build_page_map, expand_nav_and_collect, collect_clickable_elements
from tools.agent_runner.reporting import ReportGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual meaning+action run with printed outputs.")
    parser.add_argument("--url", default=os.environ.get("AGENT_URL", "https://www.cigroup.co.uk"), help="Target URL")
    parser.add_argument(
        "--goal",
        default=os.environ.get("AGENT_GOAL", "Navigate to About Us and view Clients section"),
        help="Goal for the agent",
    )
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-5-nano-2025-08-07"), help="OpenAI model")
    parser.add_argument("--headed", action="store_true", help="Run Playwright headed")
    parser.add_argument("--max-steps", type=int, default=100, help="Max agent steps before giving up")
    parser.add_argument("plan_pos", nargs="?", help="Plan file (positional)")
    parser.add_argument("--plan", help="Optional path or JSON string containing acceptance criteria")
    parser.add_argument("--scripted", action="store_true", help="Use simple heuristic navigation instead of the LLM")
    parser.add_argument("--direct", action="store_true", help="Fast direct navigation (no LLM, fastest path)")
    parser.add_argument("--hybrid", action="store_true", help="Try direct first, fall back to LLM if it fails")
    parser.add_argument("--all-stories", action="store_true", help="Run all user stories in sequence")
    parser.add_argument("--thorough", action="store_true",
                        help="Thorough mode: deep submenu scanning (slower but finds dropdowns)")
    parser.add_argument("--no-persist", action="store_true", help="Reset session between tests (default is to persist)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output (only results)")
    parser.add_argument("--report", help="Save test report to JSON file")
    return parser.parse_args()


def load_plan_arg(plan_arg: str) -> dict:
    """Load plan from a file path or a JSON string."""
    if not plan_arg:
        return {}
    plan_arg = plan_arg.strip()
    if plan_arg.startswith("{"):
        return json.loads(plan_arg)
    plan_path = os.path.abspath(plan_arg)
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_acceptance(plan: dict) -> list:
    """Pull acceptance criteria from the first user story if present."""
    try:
        stories = plan.get("user_stories") or []
        if stories and isinstance(stories[0], dict):
            return stories[0].get("acceptance_criteria") or []
    except Exception:
        return []
    return []


def extract_all_stories(plan: dict) -> list:
    """
    Get all test cases from the plan.
    Supports both formats:
    - Simple: user_stories with title, description, acceptance_criteria
    - Rich: test_cases with steps, expected_result, test_data
    """
    # Try rich format first (test_cases)
    test_cases = plan.get("test_cases")
    if test_cases:
        # Build mapping from User Stories
        tc_map = {}
        user_stories = plan.get("user_stories") or []
        for us in user_stories:
            us_id = us.get("id")
            for related_id in us.get("related_test_ids", []):
                tc_map[related_id] = us_id

        stories = []
        for tc in test_cases:
            tc_id = tc.get("id", "")
            # Inherit US ID from mapping if not explicit
            us_id = tc.get("user_story_id") or tc_map.get(tc_id, "")

            # Convert test_case to story format
            story = {
                "id": tc_id,
                "title": tc.get("title", ""),
                "description": tc.get("expected_result", ""),
                "acceptance_criteria": tc.get("steps", []),
                "test_data": tc.get("test_data", {}),
                "user_story_id": us_id,
                "feature": tc.get("feature", ""),
                "priority": tc.get("priority", ""),
                "tags": tc.get("tags", []),
            }
            stories.append(story)
        return stories

    # Fall back to simple format (user_stories)
    return plan.get("user_stories") or []


def check_acceptance(state: dict, seen_about: bool, acceptance: Optional[list] = None) -> Tuple[bool, str]:
    """
    Heuristic for the end goal: reach the Clients page after visiting About.
    """
    url = (state.get("pageUrl") or "").lower()
    text = (state.get("visibleText") or "").lower()
    # Treat headings in visible text as a signal of a clients page.
    headings = [ln.strip().lower() for ln in (state.get("visibleText") or "").splitlines() if ln.strip()]
    has_client_heading = any(ln.startswith(("clients", "our clients", "client")) for ln in headings)

    clients_page = "client" in url or has_client_heading
    clients_present = any(tok in text for tok in ("client", "clients", "our clients"))
    client_visuals = any((el.get("alt") or "").strip() for el in state.get("clickables", []) if el.get("tag") == "img")
    success = seen_about and clients_page and (clients_present or client_visuals)
    reason = []
    if acceptance:
        reason.append(f"acceptance={'; '.join(acceptance)}")
    reason.append(f"seen_about={seen_about}")
    reason.append(f"clients_page={clients_page}")
    reason.append(f"client_heading={has_client_heading}")
    reason.append(f"clients_present={clients_present}")
    reason.append(f"client_visuals={client_visuals}")
    return success, "; ".join(reason)


def choose_scripted_action(
        digest: List[Dict[str, Any]],
        elements: List[Dict[str, Any]],
        seen_about: bool,
        tried_refs: dict[int, int],
) -> Optional[Dict[str, Any]]:
    """
    Minimal heuristic navigator when --scripted is enabled.
    """

    def el_ref(el: Dict[str, Any]) -> Optional[int]:
        return el.get("ref") if el.get("ref") is not None else el.get("index")

    def el_parent(el: Dict[str, Any]) -> Optional[int]:
        return el.get("parentRef")

    def pick_from_elements(match_fn):
        for el in elements:
            if match_fn(el):
                return el
        return None

    if not seen_about:
        # Prefer header/nav region if we have y coordinate.
        about = pick_from_elements(
            lambda el: any(
                token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("about",))
                       and (el.get("boundingBox", {}).get("y") or 9999) < 400
        )
        about = about or pick_from_elements(
            lambda el: any(
                token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("about",))
        )
        if about:
            ref = el_ref(about)
            tried_refs[ref] = tried_refs.get(ref, 0) + 1
            return {"action": "click", "primary_ref": ref, "secondary_ref": None, "value": "About",
                    "notes": "Navigate to About."}

    # Single cookie accept if present.
    cookie = pick_from_elements(
        lambda el: any(token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in
                       ("accept", "cookie", "consent"))
    )
    if cookie:
        ref = el_ref(cookie)
        tried_refs[ref] = tried_refs.get(ref, 0) + 1
        if tried_refs[ref] <= 1:
            return {"action": "click", "primary_ref": ref, "secondary_ref": None, "value": "Accept",
                    "notes": "Accept/dismiss cookie banner."}

    # Try subnav child whose parent mentions About.
    candidates = [
        el
        for el in elements
        if el_parent(el) is not None
           and any(
            token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("client",))
    ]
    for client_el in candidates:
        parent_idx = el_parent(client_el)
        parent_el = next((el for el in elements if el_ref(el) == parent_idx), None)
        parent_has_about = parent_el and any(
            "about" in (parent_el.get(field) or "").lower() for field in ("text", "aria", "title"))
        if parent_el and parent_has_about and el_ref(client_el) is not None:
            child = el_ref(client_el)
            tried_refs[child] = tried_refs.get(child, 0) + 1
            return {
                "action": "hover_click",
                "primary_ref": parent_idx,
                "secondary_ref": child,
                "value": "Clients",
                "notes": "Hover About then click Clients subnav.",
            }

    clients = pick_from_elements(
        lambda el: any(
            token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("client",))
    )
    if clients:
        ref = el_ref(clients)
        tried_refs[ref] = tried_refs.get(ref, 0) + 1
        return {"action": "click", "primary_ref": ref, "secondary_ref": None, "value": "Clients",
                "notes": "Navigate to Clients."}

    return None


# EXPORTED API
async def run_test_plan(
        url: str,
        test_plan: Union[Dict[str, Any], str],
        model: str = "gpt-5-nano-2025-08-07",
        headed: bool = False,
        persist_session: bool = True,
        debug: bool = False,
        thorough: bool = False,
        hybrid: bool = True,
        max_steps: int = 100
) -> List[Dict[str, Any]]:
    """
    Run a test plan against a URL using the robust hybrid agent flow.
    """

    load_env()
    settings = Settings()
    client = openai_client()

    # Flexible input parsing
    if isinstance(test_plan, str):
        plan_data = load_plan_arg(test_plan)
    else:
        plan_data = test_plan

    stories = extract_all_stories(plan_data)
    if not stories:
        print("No user stories/test cases found in plan!")
        return []

    print(f"\n{'=' * 60}")
    print(f"RUNNING {len(stories)} TESTS (HYBRID MODE)")
    print(f"RUNNING {len(stories)} TESTS (HYBRID MODE)")
    print(f"Models: Complex={settings.model_complex}, Simple={settings.model_simple}")
    print(f"{'=' * 60}")

    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        try:
            # Initial navigation
            await page.goto(url, wait_until="networkidle", timeout=30000)

            for i, story in enumerate(stories):
                story_id = story.get("id", f"TC-{i + 1}")
                story_title = story.get("title", "Unknown")
                story_desc = story.get("description", "")
                criteria = story.get("acceptance_criteria", [])
                test_data = story.get("test_data", {})
                expected_failure = test_data.get("expected_failure", False)

                print(f"\n[{i + 1}/{len(stories)}] {story_id}: {story_title}")
                print("-" * 50)

                # SESSION PERSISTENCE check
                if i == 0 or not persist_session:
                    if page.url != url:
                        await page.goto(url, wait_until="networkidle", timeout=30000)
                    print(f"Starting from: {page.url}")
                else:
                    print(f"Continuing from: {page.url} (session persisted)")

                # Build goal
                goal = f"{story_title}. {story_desc}"
                if criteria:
                    goal += "Acceptance criteria: " + "; ".join(criteria)
                if test_data:
                    goal += f"USE THESE EXACT VALUES: {json.dumps(test_data)}"

                # INJECT TEST CONTEXT
                test_type = story.get("type", "functional")
                goal += f"\nTEST CONTEXT: This is a '{test_type}' test."
                if test_type == "negative":
                    goal += "(Expect failure/error messages. If error appears, goal is COMPLETE.)"

                # Capture initial state
                await auto_dismiss_popups(page)

                # Execution Loop
                # Use provided max_steps (default 100) as baseline, but ensure at least 5 steps per criteria item
                limit = max(max_steps, len(criteria) * 5)
                success = False
                error_msg = None
                # Initialize Observer and Memory for this story
                # Use Complex Model for critical Observation/Judgement
                observer = ObserverAgent(client, settings.model_complex)
                # Initialize Navigator (NavigatorAgent) with Complex Model
                navigator = NavigatorAgent(client, settings.model_complex)
                verifier = VerifierAgent(client, settings.model_complex)
                memory = SemanticMemory()

                steps_taken = 0
                action_history = []
                primary_ref = None
                loop_counter = 0
                failure_refs: dict[int, int] = {}
                observer_feedback = None

                for step in range(limit):
                    steps_taken = step + 1
                    state = await capture_state(page)

                    # Observer (Smart Judge) Check
                    # It validates if the goal is met OR if a bug occurred (e.g. Test expects error, App allows success)
                    obs_result = await observer.observe(
                        goal,
                        memory.get_recent_history(),
                        state.get("pageUrl", ""),
                        state.get("visibleText", "")
                    )

                    # Capture Feedback for NEXT step
                    if obs_result.get("feedback"):
                        observer_feedback = obs_result.get("feedback")

                    if debug:
                        print(f"Observer Verdict: {obs_result}")

                    if obs_result["status"] == "success":
                        print(f"Observer declared SUCCESS: {obs_result.get('reason')}")
                        success = True
                        break
                    elif obs_result["status"] == "failed":
                        print(f"Observer declared FAILURE: {obs_result.get('reason')}")
                        success = False
                        error_msg = obs_result.get("reason", "Observer declared failure")
                        break

                    # Clickable collection strategy
                    if thorough:
                        expanded_clickables = await expand_nav_and_collect(page, state.get("clickables", []))
                    else:
                        # Fast mode: try simple menu expand
                        try:
                            # Heuristic: if menu button exists, open it
                            menu_btn = page.locator("button.menu-btn, .hamburger").first
                            if await menu_btn.is_visible(timeout=500):
                                await menu_btn.click(timeout=1000)
                                await page.wait_for_timeout(300)
                        except Exception:
                            pass
                        expanded_clickables = await collect_clickable_elements(page)

                    # Build digest
                    page_map = build_page_map(state.get("rawHtml", ""), expanded_clickables, state.get("pageUrl", ""),
                                              state.get("pageTitle", ""))
                    goal_hint = " ".join(criteria) if criteria else story_title
                    digest = digest_for_action(page_map, limit=80, goal_hint=goal_hint)

                    # Augment goal with history
                    current_goal = goal
                    current_goal += "\nALWAYS re-check the page before acting: if expected URL/text is already present, declare SUCCESS instead of repeating clicks."

                    # Steer the LLM away from refs that have already failed multiple times
                    blocked_refs = [str(ref) for ref, count in failure_refs.items() if count >= 2]
                    if blocked_refs:
                        current_goal += f"\n\nAVOID these refs (failed repeatedly): {', '.join(blocked_refs)}."

                    if action_history:
                        history_str = "\n".join(action_history[-5:])
                        current_goal += f"\n\nRECENT ACTIONS (Do not repeat [SUCCESS] actions):\n{history_str}"

                    if primary_ref is None and action_history and "[FAILED]" in action_history[-1]:
                        current_goal += "\n\nWARNING: Last action failed due to missing element. RE-EXAMINE the page carefully."

                    # Inject Observer Feedback (Coaching)
                    if observer_feedback:
                        current_goal += f"\n\n[OBSERVER FEEDBACK/COACHING]: {observer_feedback}"
                        # Clear feedback after using it once so it doesn't stick forever if resolved
                        observer_feedback = None

                    # LLM Analysis
                    # Use Navigator Agent for Action Decision
                    intent = await navigator.analyze_and_act(current_goal, digest, state)
                    action = intent.get("action", "done")
                    status = intent.get("status", "running")
                    primary_ref = intent.get("primary_ref")
                    value = intent.get("value", "")

                    # DYNAMIC COMPLETION CHECK
                    if status == "success":
                        print(f"Agent declared SUCCESS status. Stopping.")
                        action_history.append(f"Step {step + 1}: Agent declared SUCCESS.")
                        success = True
                        break
                    elif status == "failed":
                        print(f"Agent declared FAILED status. Stopping.")
                        action_history.append(f"Step {step + 1}: Agent declared FAILURE.")
                        success = False
                        error_msg = "Agent determined goal is impossible or failed."
                        break
                    elif action == "done" and status == "running":
                        print("  Agent said 'done' (legacy). Stopping.")
                        success = True
                        break

                    # Record intention
                    action_desc = f"Step {step + 1}: {action} {value if value else ''} (Ref {primary_ref})"

                    # STATIONARY LOOP CHECK
                    is_looping = False
                    if len(action_history) >= 3:
                        last_3 = action_history[-3:]
                        # Extract core action string (ignoring Step X prefix)
                        current_core = f"{action} {value if value else ''} (Ref {primary_ref})"

                        # Check if all last 3 contain this exact action
                        # We use a loose check because previous history entries have [SUCCESS]/[FAILED] suffixes
                        match_count = 0
                        for hist in last_3:
                            if current_core in hist:
                                match_count += 1

                        if match_count >= 3:
                            print(
                                f"WARNING: Detected stationary loop (repeated action '{current_core}' 3 times). Aborting step.")
                            action_history.append(f"Step {step + 1}: [FAILED] Loop detected. Aborted.")
                            if primary_ref is not None:
                                failure_refs[primary_ref] = failure_refs.get(primary_ref, 0) + 1

                            # HARD STOP LOGIC
                            if loop_counter >= 5:
                                print("CRITICAL: Infinite Loop Spam Detected. Hard Stopping Test.")
                                success = False
                                error_msg = "Infinite Loop Detected"
                                break
                            loop_counter += 1
                            continue  # Skip waiting / applying
                        else:
                            loop_counter = 0  # Reset if valid action
                    else:
                        loop_counter = 0  # Reset if not enough history to check for loop

                    action_history.append(action_desc)

                    if debug:
                        print(f"Step {step + 1}: {action} ref={primary_ref} - {intent.get('notes', '')[:50]}...")

                    if action == "done":
                        print("LLM says goal complete.")
                        # Verify logic
                        page_text = await page.inner_text("body")
                        final_url = page.url

                        success, reason = await verifier.verify(
                            story,
                            final_url,
                            page_text.lower()
                        )
                        if debug or not success:
                            print(f"Verification: {reason}")

                        break

                    # Apply Action (Robust)
                    action_success = False
                    max_retries = 3

                    for attempt in range(max_retries):
                        if primary_ref is None and action not in ["done", "goto"]:
                            print("Action missing primary_ref, skipping retry and re-digesting...")
                            action_success = False
                            break

                        try:
                            if await apply_action(page, intent, expanded_clickables):
                                action_success = True
                                if debug: print(f"  Action applied: {action}")
                                break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                if debug: print(f"  Action failed: {e}")
                                # Screenshot on final failure
                                try:
                                    os.makedirs("tests/failures", exist_ok=True)
                                    ts = int(asyncio.get_event_loop().time())
                                    path = f"tests/failures/{story_id}_fail_{ts}.png"
                                    await page.screenshot(path=path)
                                    if debug: print(f"  Saved failure screenshot: {path}")
                                except:
                                    pass
                            else:
                                if debug: print(f"Action error (attempt {attempt + 1}): {e}. Retrying...")
                                await asyncio.sleep(attempt + 1)

                    if action_success:
                        action_history[-1] += " [SUCCESS]"
                        # Log to Semantic Memory
                        memory.add_state(
                            step=step + 1,
                            url=page.url,
                            action=f"{action} {primary_ref if primary_ref else ''} ({intent.get('notes', '')})",
                        )
                    else:
                        action_history[-1] += " [FAILED]"
                        if primary_ref is not None:
                            failure_refs[primary_ref] = failure_refs.get(primary_ref, 0) + 1
                        if primary_ref is None:
                            pass
                        else:
                            continue

                    await asyncio.sleep(0.5)

                if not success:
                    if not error_msg:
                        error_msg = "Max steps reached"
                    print(f"  Goal failed ({error_msg})")

                    # INVESTIGATION MODE
                    print(f"[INVESTIGATION] Entering Investigation Mode (5 steps) to diagnose: {error_msg}")
                    inv_goal = f"TEST FAILED: {error_msg}. Your job is to INVESTIGATE THE FAILURE. Do NOT retry the failing action blindly. You have 5 steps to explore the app (e.g. check Cart, History, Profile) or try alternative navigation to confirm if the bug is persistent or page-specific. Report findings."

                    inv_findings = []
                    # Use COMPLEX model for Investigation (needs smarts to explore)
                    inv_observer = ObserverAgent(client, settings.model_complex)
                    # Use Navigator for Investigation actions
                    inv_navigator = NavigatorAgent(client, settings.model_complex)

                    for inv_step in range(5):
                        state = await capture_state(page)

                        # Use COMPLEX model to analyze investigation step
                        digest_inv = digest_for_action(
                            build_page_map(state.get("rawHtml", ""), await collect_clickable_elements(page),
                                           state.get("pageUrl", ""), state.get("pageTitle", "")),
                            limit=80, goal_hint="Investigate failure"
                        )

                        inv_intent = await inv_navigator.analyze_and_act(inv_goal, digest_inv, state)
                        inv_action = inv_intent.get("action", "done")
                        inv_note = inv_intent.get("notes", "")

                        print(f"[Inv Step {inv_step + 1}] {inv_action} - {inv_note}")
                        inv_findings.append(f"{inv_action} - {inv_note}")

                        if inv_action == "done":
                            break

                        await apply_action(page, inv_intent, await collect_clickable_elements(page))
                        await asyncio.sleep(1)

                    if inv_findings:
                        # SUMMARISE FINDINGS (Simple Model)
                        summarizer = SummariserAgent(client, settings.model_simple)
                        narrative = await summarizer.summarize_investigation(inv_findings, error_msg)
                        error_msg += f" || INVESTIGATION FINDINGS: {narrative}"

                result = {
                    "id": story_id,
                    "title": story_title,
                    "status": "PASS" if success else "FAIL",
                    "error": error_msg,
                    "user_story_id": story.get("user_story_id", ""),
                    "priority": story.get("priority", "P1"),
                    "steps": story.get("acceptance_criteria", []),
                    "expected_result": story.get("description", "")
                }
                results.append(result)

            # Summary
            print("\n" + "=" * 50)
            print("TEST RESULTS SUMMARY")
            print("=" * 50)
            passed = 0

            # Group by User Story ID
            from itertools import groupby
            results.sort(key=lambda x: x.get("user_story_id", "") or "")

            for us_id, group_iter in groupby(results, key=lambda x: x.get("user_story_id", "") or ""):
                group = list(group_iter)
                if us_id:
                    us_passed = all(r["status"] == "PASS" for r in group)
                    icon = "[PASS]" if us_passed else "[FAIL]"
                    print(f"\n{icon} User Story: {us_id}")
                    for res in group:
                        status = res.get("status")
                        icon = "[PASS]" if status == "PASS" else "[FAIL]"
                        error = f" ({res.get('error')})" if status == "FAIL" and res.get('error') is not None else ""
                        print(f"  {icon} {res.get('id')}: {res.get('title')} - {status}{error}")
                else:
                    for res in group:
                        status = res.get("status")
                        icon = "[PASS]" if status == "PASS" else "[FAIL]"
                        error = f" ({res.get('error')})" if status == "FAIL" and res.get('error') is not None else ""
                        print(f"{icon} {res.get('id')}: {res.get('title')} - {status}{error}")

                passed += sum(1 for r in group if r["status"] == "PASS")

            print("-" * 50)
            print(f"Total: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}")

        finally:
            await browser.close()

    return results


async def main_async():
    load_env()
    args = parse_args()

    # Run loop
    results = await run_test_plan(
        url=args.url,
        test_plan=args.plan_pos or args.plan,
        model=args.model,
        headed=args.headed,
        persist_session=not args.no_persist,
        debug=args.debug,
        thorough=args.thorough,
        hybrid=args.hybrid,
        max_steps=args.max_steps
    )

    if args.report:
        try:
            # Load plan data for report context (Shared)
            plan_data = {}
            if args.plan:
                plan_data = load_plan_arg(args.plan)
            elif args.plan_pos:
                plan_data = load_plan_arg(args.plan_pos)

            generator = ReportGenerator(results, plan_data)

            if args.report.endswith(".md"):
                # Generate Markdown Report
                content = generator.generate_markdown()
                with open(args.report, "w") as f:
                    f.write(content)
            elif args.report.endswith(".xlsx"):
                # Generate Excel Report
                generator.generate_xlsx(args.report)
            else:
                # Default JSON
                with open(args.report, "w") as f:
                    json.dump(results, f, indent=2)
            print(f"Report saved to {args.report}")
        except Exception as e:
            print(f"Failed to save report: {e}")

    # Exit code
    passed = sum(1 for r in results if r["status"] == "PASS")
    if passed < len(results):
        exit(1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
