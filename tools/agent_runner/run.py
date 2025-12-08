"""
Manual driver to run meaning + action agents and print their outputs.
Requires OPENAI_API_KEY and network access. Not part of automated tests.

Usage:
  PYTHONPATH=. python tools/agent_runner/run.py \
    --url https://www.cigroup.co.uk \
    --goal "Navigate to About Us and view Clients section" \
    --model gpt-5-nano-2025-08-07
"""
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Union, Optional, Tuple

from playwright.async_api import async_playwright

from tools.agent_runner.agents.combined_agent import analyze_and_act
from tools.agent_runner.agents.meaning_agent import extract_meaning
from tools.agent_runner.page_model import build_page_map, expand_nav_and_collect, human_processable, \
    collect_clickable_elements
from tools.agent_runner.actions import (
    apply_action,
    auto_dismiss_popups,
    capture_state,
    digest_for_action,
    load_env,
    openai_client,
)


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
    parser.add_argument("--max-steps", type=int, default=15, help="Max agent steps before giving up")
    parser.add_argument("plan_pos", nargs="?", help="Plan file (positional)")
    parser.add_argument("--plan", help="Optional path or JSON string containing acceptance criteria")
    parser.add_argument("--scripted", action="store_true", help="Use simple heuristic navigation instead of the LLM")
    parser.add_argument("--direct", action="store_true", help="Fast direct navigation (no LLM, fastest path)")
    parser.add_argument("--hybrid", action="store_true", help="Try direct first, fall back to LLM if it fails")
    parser.add_argument("--all-stories", action="store_true", help="Run all user stories in sequence")
    parser.add_argument("--thorough", action="store_true",
                        help="Thorough mode: deep submenu scanning (slower but finds dropdowns)")
    parser.add_argument("--persist", action="store_true", help="Keep session between tests (don't reset to homepage)")
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
        stories = []
        for tc in test_cases:
            # Convert test_case to story format
            story = {
                "id": tc.get("id", ""),
                "title": tc.get("title", ""),
                "description": tc.get("expected_result", ""),
                "acceptance_criteria": tc.get("steps", []),
                "test_data": tc.get("test_data", {}),
                "feature": tc.get("feature", ""),
                "priority": tc.get("priority", ""),
                "tags": tc.get("tags", []),
            }
            stories.append(story)
        return stories

    # Fall back to simple format (user_stories)
    return plan.get("user_stories") or []


def check_acceptance(state: dict, seen_about: bool, acceptance: list | None = None) -> Tuple[bool, str]:
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


async def run_story_direct(page, story: dict) -> bool:
    """
    Fast direct navigation based on story keywords.
    """
    story_id = story.get("id", "Unknown")
    story_title = story.get("title", "")
    criteria = story.get("acceptance_criteria", [])
    criteria_text = " ".join(criteria).lower()

    print(f"\n=== DIRECT MODE: {story_id} - {story_title} ===")

    # Click menu button if present
    menu_selectors = ["button.menu-btn", "button[aria-label*='menu' i]", ".hamburger"]
    for sel in menu_selectors:
        try:
            menu_btn = page.locator(sel).first
            if await menu_btn.is_visible(timeout=500):
                await menu_btn.click(timeout=1000)
                await page.wait_for_timeout(500)
                break
        except Exception:
            continue

    # Determine navigation path based on story keywords
    nav_targets = []
    if "client" in criteria_text:
        nav_targets = [("About Us", None), ("Our Clients", "client")]
    elif "work" in criteria_text or "portfolio" in criteria_text:
        nav_targets = [("Work", "work")]
    elif "insight" in criteria_text or "blog" in criteria_text:
        nav_targets = [("Insights", "insight")]
    elif "people" in criteria_text or "team" in criteria_text:
        nav_targets = [("About Us", None), ("Our People", "people")]
    elif "contact" in criteria_text or "form" in criteria_text:
        nav_targets = [("Let's talk", None)]  # Form is on homepage, just scroll
    elif "social" in criteria_text:
        nav_targets = []  # Social links are always visible

    # Execute navigation
    for target_text, url_check in nav_targets:
        print(f"Clicking: '{target_text}'...")
        try:
            link = page.get_by_text(target_text, exact=False).first
            await link.click(timeout=3000)
            await page.wait_for_timeout(1000)
        except Exception as e:
            print(f"Failed to click {target_text}: {e}")
            return False

    # Verify based on story type
    final_url = page.url.lower()
    visible_text = ""
    try:
        visible_text = (await page.evaluate("() => document.body.innerText || ''")).lower()
    except Exception:
        pass

    success = False
    if "client" in criteria_text:
        success = "client" in final_url
    elif "work" in criteria_text:
        success = "work" in final_url
    elif "insight" in criteria_text:
        success = "insight" in final_url
    elif "people" in criteria_text:
        success = "people" in final_url
    elif "contact" in criteria_text or "form" in criteria_text:
        success = "email" in visible_text or "enquiry" in visible_text or "send" in visible_text
    elif "social" in criteria_text:
        success = "linkedin" in visible_text or "twitter" in visible_text or "instagram" in visible_text
    else:
        success = True  # Default to pass if no specific check

    print(f"Result: {'PASS' if success else 'FAIL'}")
    return success


# EXPORTED API
async def run_test_plan(
        url: str,
        test_plan: Union[Dict[str, Any], str],
        model: str = "gpt-5-nano-2025-08-07",
        headed: bool = False,
        persist_session: bool = False,
        debug: bool = False,
        thorough: bool = False,
        hybrid: bool = True
) -> List[Dict[str, Any]]:
    """
    Run a test plan against a URL using the robust hybrid agent flow.
    """
    load_env()
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
                    goal += " Acceptance criteria: " + "; ".join(criteria)
                if test_data:
                    goal += f" USE THESE EXACT VALUES: {json.dumps(test_data)}"
                
                # Capture initial state
                await auto_dismiss_popups(page)
                
                # Execution Loop
                max_steps = max(len(criteria) * 5, 25)
                success = False
                error_msg = None
                steps_taken = 0
                action_history = []

                for step in range(max_steps):
                    steps_taken = step + 1
                    state = await capture_state(page)

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
                    page_map = build_page_map(state.get("rawHtml", ""), expanded_clickables, state.get("pageUrl", ""), state.get("pageTitle", ""))
                    goal_hint = " ".join(criteria) if criteria else story_title
                    digest = digest_for_action(page_map, limit=80, goal_hint=goal_hint)

                    # Augment goal with history
                    current_goal = goal
                    if action_history:
                        history_str = "\\n".join(action_history[-5:])
                        current_goal += f"\\n\\nRECENT ACTIONS:\\n{history_str}"

                    # LLM Analysis
                    intent = await analyze_and_act(client, model, current_goal, digest, state)
                    action = intent.get("action", "done")
                    primary_ref = intent.get("primary_ref")
                    value = intent.get("value", "")
                    
                    # Record intention
                    action_desc = f"Step {step+1}: {action} {value if value else ''} (Ref {primary_ref})"
                    action_history.append(action_desc)
                    
                    if debug:
                        print(f"  Step {step + 1}: {action} ref={primary_ref} - {intent.get('notes', '')[:50]}...")

                    if action == "done":
                        print(f"  ✅ LLM says goal complete!")
                        # Verify logic
                        page_text = await page.inner_text("body")
                        final_url = page.url
                        
                        if expected_failure:
                             # Negative test
                            expected_text = test_data.get("expected_text", "")
                            if expected_text and expected_text.lower() in page_text.lower():
                                success = True
                        else:
                            # Positive test
                            expected_url = test_data.get("expected_url", "")
                            expected_text = test_data.get("expected_text", "")
                            
                            url_match = not expected_url or expected_url in final_url
                            text_match = not expected_text or expected_text.lower() in page_text.lower()
                            
                            if url_match and text_match:
                                success = True
                            else:
                                if not url_match: print(f"  ❌ URL mismatch: expected '{expected_url}' not in '{final_url}'")
                                if not text_match: print(f"  ❌ Text mismatch: expected '{expected_text}' not found")
                        break

                    # Apply Action (Robust)
                    action_success = False
                    max_retries = 3
                    
                    for attempt in range(max_retries):
                        try:
                            if await apply_action(page, intent, expanded_clickables):
                                action_success = True
                                if debug: print(f"  ✅ Action applied: {action}")
                                break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                if debug: print(f"  ❌ Action failed: {e}")
                                # Screenshot on final failure
                                try:
                                    os.makedirs("tests/failures", exist_ok=True)
                                    ts = int(asyncio.get_event_loop().time())
                                    path = f"tests/failures/{story_id}_fail_{ts}.png"
                                    await page.screenshot(path=path)
                                    if debug: print(f"  📸 Saved failure screenshot: {path}")
                                except: pass
                            else:
                                if debug: print(f"  ⚠️ Action error (attempt {attempt+1}): {e}. Retrying...")
                                await asyncio.sleep(attempt + 1)
                                

                    
                    if not action_success:
                        continue
                    
                    await asyncio.sleep(0.5)

                if not success:
                    if not error_msg:
                        error_msg = "Max steps reached"
                    print(f"  ❌ Goal failed ({error_msg})")

                result = {
                    "id": story_id,
                    "title": story_title,
                    "status": "PASS" if success else "FAIL",
                    "error": error_msg
                }
                results.append(result)

            # Summary
            print("\n" + "="*50)
            print("TEST RESULTS SUMMARY")
            print("="*50)
            passed = 0
            for res in results:
                status = res.get("status")
                icon = "✅" if status == "PASS" else "❌"
                print(f"{icon} {res.get('id')}: {res.get('title')} - {status}")
                if status == "PASS": passed += 1

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
        persist_session=args.persist,
        debug=args.debug,
        thorough=args.thorough,
        hybrid=args.hybrid
    )

    if args.report:
        try:
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
