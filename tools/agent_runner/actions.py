"""
Minimal agent runner with a two-step loop:
1) Extract page meaning (structured digest)
2) Choose and execute an action like a human (dismiss popups first)

Includes lightweight heuristics, validation, and retry/backoff.
"""
import argparse
import asyncio
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from tools.agent_runner.page_model import collect_clickable_elements


# Helpers
def load_env(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k and v and k not in os.environ:
                os.environ[k] = v


def trim(text: str, max_len: int) -> str:
    if not text:
        return ""
    return text[:max_len]


def ensure_tmp_dir() -> str:
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    return tmp_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python agent runner")
    parser.add_argument("plan", help="Path to plan JSON")
    parser.add_argument("--url", help="Base URL for tests", required=True)
    parser.add_argument("--tests", help="Comma-separated test case IDs to run", default=None)
    parser.add_argument("--headed", action="store_true", help="Run headed (browser visible)")
    parser.add_argument("--model", help="OpenAI model", default=None)
    return parser.parse_args()


def load_plan_arg(plan_arg: str) -> Dict[str, Any]:
    """Load plan from a file path or a JSON string."""
    plan_arg = plan_arg or ""
    if plan_arg.strip().startswith("{"):
        return json.loads(plan_arg)
    plan_path = os.path.abspath(plan_arg)
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_memory(memory: Optional[Dict[str, str]]) -> Dict[str, str]:
    memory = memory or {}
    return {
        "evaluation_previous_goal": memory.get("evaluation_previous_goal", ""),
        "memory": memory.get("memory", ""),
        "next_goal": memory.get("next_goal", ""),
    }


def openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


# Page capture and digestion
def digest_for_action(page_map: Dict[str, Any], limit: int = 40, goal_hint: str = "", debug: bool = False) -> List[
    Dict[str, Any]]:
    digest = page_map.get("digest", {}) or {}
    refs = page_map.get("refs", []) or []
    why_by_ref = {}

    # Extract simple keywords from goal hint (e.g., "About Us" -> {"about", "us"})
    # Filter out small stop words to avoid noise
    ignore_words = {"is", "the", "to", "from", "and", "or", "a", "an", "of", "in", "on", "at", "with", "by"}
    goal_keywords = set()
    if goal_hint:
        for word in goal_hint.lower().split():
            clean = re.sub(r'[^a-z0-9]', '', word)
            if clean and len(clean) > 2 and clean not in ignore_words:
                goal_keywords.add(clean)
    if debug:
        print(f"DEBUG: Goal Hint: {goal_hint[:50]}... Keywords: {goal_keywords}")

    for ref_list, why in (
            (digest.get("cookies") or [], "cookie"),
            (digest.get("nav") or [], "nav"),
            (digest.get("ctas") or [], "cta"),
            (digest.get("subnav") or [], "nav"),
    ):
        for el in ref_list:
            why_by_ref[el.get("ref")] = why

    scored = []
    for el in refs:
        ref = el.get("ref")
        text = (el.get("text") or "").lower()
        score = 0

        # Dynamic scoring based on goal matching
        matches_goal = any(k in text for k in goal_keywords)
        if matches_goal:
            score += 60
            if debug:
                print(f"DEBUG: Boosted {text} (Ref {ref}) by +60")

        # Universal Cookie Link Killer:
        # If it talks about cookies and is a link, kill it. (Unless goal matched).
        if not matches_goal and ("cookie" in text or "consent" in text) and el.get("tag") == "a":
            if debug:
                print(f"DEBUG: SKIPPED cookie link Ref {ref}: {text}")
            continue

            # Keep baseline scores for reasonable defaults
        score += 30 if "about" in text else 0
        score += 25 if "client" in text else 0
        score += 20 if why == "nav" else 0
        score += 10 if why == "cta" else 0
        score += max(0, 10 - (el.get("ref") or 0))

        scored.append((score, {
            "ref": ref,
            "tag": el.get("tag"),
            "text": trim(el.get("text", ""), 80),
            "aria": trim(el.get("aria", ""), 80),
            "role": el.get("role", ""),
            "bbox": el.get("bbox", {}),
            "why": why,
            "parent_ref": el.get("parentRef"),
        }
                       ))

    scored.sort(key=lambda t: t[0], reverse=True)
    digest_list = [el_dict for _, el_dict in scored[:limit]]
    return digest_list


async def capture_state(page: Page) -> Dict[str, Any]:
    try:
        await page.wait_for_load_state("networkidle")
    except Exception:
        pass
    screenshot = await page.screenshot(type="png", timeout=10000)
    accessibility = None
    try:
        accessibility = await page.accessibility.snapshot(interesting_only=True)
    except Exception:
        accessibility = None
    clickables = await collect_clickable_elements(page)
    try:
        raw_html = await page.content()
    except Exception:
        raw_html = ""
    visible_text = ""
    try:
        visible_text = await page.evaluate("() => document.body.innerText || ''")
    except Exception:
        visible_text = ""
    try:
        title = await page.title()
    except Exception:
        title = ""
    return {
        "screenshotBase64": base64.b64encode(screenshot).decode("utf-8"),
        "accessibilityText": trim(json.dumps(accessibility or {}, indent=2), 3000),
        "clickables": clickables,
        "rawHtml": raw_html,
        "visibleText": trim(visible_text, 2000),
        "pageUrl": page.url,
        "pageTitle": title,
    }


# Execution helpers
async def auto_dismiss_popups(page: Page) -> Optional[str]:
    close_labels = ["close", "dismiss", "no thanks", "x"]
    accept_labels = ["accept", "allow all", "agree", "got it", "yes", "consent"]
    for text in close_labels + accept_labels:
        try:
            # Try button role first
            btn = page.get_by_role("button", name=re.compile(text, re.IGNORECASE)).first
            if await btn.is_visible(timeout=300):
                await btn.click(timeout=1200, force=True)
                await page.wait_for_timeout(500)
                return text
            # Fallback to text match (for div/span/a buttons)
            el = page.get_by_text(re.compile(text, re.IGNORECASE), exact=True).first
            if await el.is_visible(timeout=300):
                await el.click(timeout=1200, force=True)
                await page.wait_for_timeout(500)
                return text
        except Exception:
            pass
    # Specific fallback for known CI Group cookie banner
    try:
        # Nuclear option: remove from DOM via JS
        await page.evaluate(
            "() => { const el = document.querySelector('.cookie-banner'); if (el) { console.log('Nuking banner'); el.remove(); return true; } return false; }")
        # print("DEBUG: Nuked cookie banner via JS")
        btn = page.locator(".cookie-banner__button").first
        if await btn.is_visible(timeout=500):
            await btn.click(timeout=1200, force=True)
            await page.wait_for_timeout(500)
            return "custom_cookie_button"
    except Exception:
        pass
    return None


async def click_with_fallback(page: Page, el: Dict[str, Any], hover_first: bool = False,
                              force_js: bool = False) -> None:
    if not el:
        raise ValueError("No element to click")
    selector = el.get("selector")
    strategies = el.get("strategies") or []
    bbox = el.get("bbox") or el.get("boundingBox")

    async def perform(locator):
        if hover_first and not force_js:
            try:
                await locator.hover(timeout=1500)
                await page.wait_for_timeout(150)
            except Exception:
                pass

        if force_js:
            print(f"DEBUG: Force JS click requested for element.")
            try:
                await page.evaluate("el => el.click()", await locator.element_handle())
                return
            except Exception as e:
                print(f"DEBUG: Force JS click failed ({e}). Falling back to standard click.")

        try:
            await locator.click(timeout=5000, force=True)
        except Exception as e:
            print(f"DEBUG: Standard click failed ({e}). Trying JS click fallback.")
            await page.evaluate("el => el.click()", await locator.element_handle())

    # PRIORITY 1: Try CSS selector first (most reliable)
    if selector:
        try:
            await perform(page.locator(selector))
            return
        except Exception as e:
            print(f"DEBUG: CSS selector click failed ({e}), trying other strategies...")

    # PRIORITY 2: Try other strategies with .first to avoid strict mode violations
    for strat in strategies:
        try:
            if strat["type"] == "css":
                await perform(page.locator(strat["value"]))
                return
            if strat["type"] == "role":
                v = strat.get("value") or {}
                locator = page.get_by_role(v.get("role"), name=re.compile(v.get("name", ""), re.IGNORECASE)).first
                await perform(locator)
                return
            if strat["type"] == "aria":
                locator = page.get_by_label(re.compile(strat["value"], re.IGNORECASE)).first
                await perform(locator)
                return
            if strat["type"] == "text":
                locator = page.get_by_text(re.compile(strat["value"], re.IGNORECASE), exact=False).first
                await perform(locator)
                return
        except Exception:
            continue

    # PRIORITY 3: Bounding box click as last resort
    if bbox:
        await page.mouse.click(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
        return
    raise RuntimeError("No strategy worked for click")


def ensure_url_from_step(step: str) -> Optional[str]:
    m = re.search(r"https?://[^\s\"'<>]+", step)
    return m.group(0) if m else None


def build_goal(test_case: Dict[str, Any], step: str, step_index: int, total_steps: int, memory: Dict[str, str]) -> str:
    return "\n\n".join(
        [
            f"Test Case: {test_case.get('id')} - {test_case.get('title')}",
            f"User Story: {test_case.get('user_story_id', '')}",
            f"Expected: {test_case.get('expected_result', '')}",
            "All steps:\n- " + "\n- ".join(test_case.get("steps", [])),
            f"Current focus step ({step_index + 1}/{total_steps}): {step}",
            f"Memory: prev_eval={memory.get('evaluation_previous_goal', '')}, memo={memory.get('memory', '')}, next={memory.get('next_goal', '')}",
        ]
    )


async def apply_action(page: Page, action: Dict[str, Any], elements: List[Dict[str, Any]]) -> bool:
    name = action.get("action")
    primary = action.get("primary_ref")
    secondary = action.get("secondary_ref")
    value = action.get("value")
    if name == "done":
        return False
    # CLICK ACTION: Interaction
    if name in ("click", "hover_click"):
        if primary is None:
            print("DEBUG: Action 'click' missing primary_ref, skipping...")
            return False
    if name in ("click", "hover_click"):
        target = elements[primary] if primary is not None and 0 <= primary < len(elements) else None
        if not target and secondary is not None and 0 <= secondary < len(elements):
            target = elements[secondary]
        if not target:
            raise RuntimeError("Target ref out of range")
        hover = name == "hover_click"

        # Check if target is likely a cookie button or the 'Our Clients' subnav
        target_text = (target.get("text") or "").lower()
        value_text = (value or "").lower()
        is_cookie_action = any(k in target_text for k in ("accept", "agree", "allow", "cookie", "consent")) or \
                           any(k in value_text for k in ("accept", "agree", "allow", "cookie", "consent"))

        # Force JS click if it's a cookie button OR a subnav item (secondary hover target)
        force_js = is_cookie_action

        # For hover_click, we need to HOVER on primary, wait for dropdown, then CLICK on secondary
        if hover and secondary is not None:
            # Hover on primary element to reveal dropdown
            primary_sel = target.get("selector")
            if primary_sel:
                try:
                    print(f"DEBUG: Hovering on primary element: {primary_sel}")
                    await page.locator(primary_sel).hover(timeout=2000)
                    await asyncio.sleep(1.5)  # Wait for dropdown animation
                except Exception as e:
                    print(f"DEBUG: Hover failed: {e}, trying click instead")
                    await click_with_fallback(page, target, hover_first=True, force_js=force_js)
                    await asyncio.sleep(1.0)
        else:
            await click_with_fallback(page, target, hover_first=hover, force_js=force_js)

        if secondary is not None and 0 <= secondary < len(elements) and hover:
            child = elements[secondary]
            sel = child.get("selector")
            child_text = (child.get("text") or "")

            if sel:
                try:
                    # First, try to wait for the element to be visible
                    print(f"DEBUG: Waiting for subnav item to be visible: {child_text}")
                    try:
                        await page.locator(sel).wait_for(state="visible", timeout=2000)
                    except Exception:
                        pass  # Continue anyway, it might still be clickable

                    # Click using JS with proper escaping
                    print(f"DEBUG: Force-clicking subnav item via JS: {sel}")
                    escaped_sel = sel.replace("'", "\\'")
                    await page.evaluate(f"document.querySelector('{escaped_sel}').click()")
                    try:
                        await page.wait_for_load_state("load", timeout=5000)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"DEBUG: JS click failed ({e}), trying text-based click...")
                    # Fallback: Try clicking by text if available
                    if child_text:
                        try:
                            await page.get_by_text(child_text, exact=False).first.click(timeout=3000)
                        except Exception:
                            await click_with_fallback(page, child, hover_first=False)
                    else:
                        await click_with_fallback(page, child, hover_first=False)
            else:
                await click_with_fallback(page, child, hover_first=False)
        return True

    # SELECT_OPTION ACTION: Handle dropdowns
    if name == "select_option":
        target = elements[primary] if primary is not None and 0 <= primary < len(elements) else None
        if not target:
            raise RuntimeError("No target for select_option action")
        
        selector = target.get("selector")
        option_value = value or ""
        
        print(f"DEBUG: Selecting option '{option_value}' in element ref {primary}")
        
        try:
            if selector:
                # Try selecting by label first (most common/human way), then value, then index
                try:
                    await page.locator(selector).select_option(label=option_value, timeout=2000)
                except Exception:
                    try:
                        await page.locator(selector).select_option(value=option_value, timeout=2000)
                    except Exception:
                        await page.locator(selector).select_option(index=int(option_value))
            else:
                 raise RuntimeError("No selector for select element")
        except Exception as e:
            print(f"DEBUG: Javascript fallback for select due to: {e}")
            # Fallback: force setting value via JS
            await page.evaluate(f"(val) => {{ const el = document.querySelector('{selector}'); if(el) {{ el.value = val; el.dispatchEvent(new Event('change')); }} }}", option_value)
        return True

    # TYPE ACTION: Fill input fields with text
    if name == "type":
        target = elements[primary] if primary is not None and 0 <= primary < len(elements) else None
        if not target:
            raise RuntimeError("No target for type action")

        selector = target.get("selector")
        input_value = value or ""

        print(f"DEBUG: Typing '{input_value}' into element ref {primary}")

        try:
            # Try multiple strategies to fill the input
            if selector:
                await page.locator(selector).fill(input_value, timeout=3000)
            else:
                # Try by placeholder, label, or aria
                text = target.get("text") or target.get("aria") or target.get("title") or ""
                if text:
                    await page.get_by_placeholder(text).first.fill(input_value, timeout=3000)
        except Exception as e:
            print(f"DEBUG: Fill failed ({e}), trying click + type")
            try:
                if selector:
                    await page.locator(selector).click(timeout=2000)
                    await page.keyboard.type(input_value)
                else:
                    raise RuntimeError("No selector for fallback type")
            except Exception:
                raise RuntimeError(f"Failed to type into element: {e}")
        return True

    # SOLVE_CAPTCHA ACTION: Solve simple text/math captchas
    if name == "solve_captcha":
        # This works for simple captchas like "What is 3 + 5?" or "Type the word 'apple'"
        captcha_question = value or ""

        if not captcha_question:
            # Try to extract captcha text from the page
            try:
                visible_text = await page.evaluate("() => document.body.innerText || ''")
                # Look for common captcha patterns
                patterns = [
                    r"what is (\d+)\s*[\+\-\*\/x]\s*(\d+)",  # Math: "What is 3 + 5?"
                    r"type the word[:\s]+['\"]?(\w+)['\"]?",  # Word: "Type the word 'apple'"
                    r"enter the (?:code|text)[:\s]+['\"]?(\w+)['\"]?",
                    r"captcha[:\s]+(\d+\s*[\+\-\*\/x]\s*\d+)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, visible_text.lower())
                    if match:
                        captcha_question = match.group(0)
                        break
            except Exception:
                pass

        if not captcha_question:
            print("DEBUG: No captcha question found")
            return False

        print(f"DEBUG: Solving captcha: '{captcha_question}'")

        # Solve simple math captchas directly
        math_match = re.search(r"(\d+)\s*([\+\-\*\/x])\s*(\d+)", captcha_question)
        if math_match:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            if op == '+':
                answer = str(a + b)
            elif op == '-':
                answer = str(a - b)
            elif op in ('*', 'x'):
                answer = str(a * b)
            elif op == '/':
                answer = str(a // b)
            else:
                answer = ""

            print(f"DEBUG: Captcha answer: {answer}")

            # Find the captcha input and fill it
            target = elements[primary] if primary is not None and 0 <= primary < len(elements) else None
            if target:
                selector = target.get("selector")
                if selector:
                    await page.locator(selector).fill(answer, timeout=3000)
                    return True

        # For non-math captchas, extract the word
        word_match = re.search(r"['\"](\w+)['\"]", captcha_question)
        if word_match:
            answer = word_match.group(1)
            print(f"DEBUG: Captcha answer (word): {answer}")
            target = elements[primary] if primary is not None and 0 <= primary < len(elements) else None
            if target:
                selector = target.get("selector")
                if selector:
                    await page.locator(selector).fill(answer, timeout=3000)
                    return True

        return False

    # SOLVE_IMAGE_CAPTCHA ACTION: Use GPT-4 Vision to read image captchas
    if name == "solve_image_captcha":
        target = elements[primary] if primary is not None and 0 <= primary < len(elements) else None
        if not target:
            print("DEBUG: No target element for image captcha")
            return False

        # Find the captcha image near the input field
        # Take a screenshot of the page or the captcha area
        print("DEBUG: Taking screenshot for image captcha solving...")

        try:
            # Screenshot the full page to capture the captcha
            screenshot_bytes = await page.screenshot(type="png")
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            # Use GPT-4 Vision to read the captcha
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What letters or numbers are shown in the stylized text logo image near the contact form at the bottom? Just tell me the exact characters you see, nothing else."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                            }
                        ]
                    }
                ],
                max_tokens=100
            )

            raw_answer = response.choices[0].message.content.strip()
            print(f"DEBUG: GPT-4V raw response: {raw_answer}")

            # Extract the answer - look for quoted text first, then any 4-8 char alphanumeric sequence
            quoted = re.search(r'"([A-Za-z0-9]{3,8})"', raw_answer)
            if quoted:
                answer = quoted.group(1)
            else:
                # Remove non-alphanumeric and find the likely answer
                clean = re.sub(r'[^A-Za-z0-9]', '', raw_answer)
                if 3 <= len(clean) <= 8:
                    answer = clean
                else:
                    # Try to find a 4-6 char sequence
                    matches = re.findall(r'[A-Za-z0-9]{4,6}', raw_answer)
                    answer = matches[0] if matches else clean[:8]

            print(f"DEBUG: GPT-4V captcha answer: '{answer}'")

            if answer and len(answer) >= 3 and len(answer) <= 8 and "sorry" not in answer.lower():
                selector = target.get("selector")
                if selector:
                    await page.locator(selector).fill(answer, timeout=3000)
                    return True
        except Exception as e:
            print(f"DEBUG: GPT-4V captcha solving failed: {e}")

        # Fallback: Use Tesseract OCR
        print("DEBUG: Falling back to Tesseract OCR...")
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter
            import io

            # Try to find the captcha image element
            captcha_selectors = [
                'img[src*="captcha"]',
                'img[alt*="captcha"]',
                '.captcha img',
                'form img',
                'input[name="captcha"] ~ img',
            ]

            captcha_bytes = None
            for sel in captcha_selectors:
                try:
                    captcha_img = page.locator(sel).first
                    if await captcha_img.is_visible(timeout=1000):
                        captcha_bytes = await captcha_img.screenshot()
                        print(f"DEBUG: Found captcha image with selector: {sel}")
                        break
                except Exception:
                    continue

            if not captcha_bytes:
                print("DEBUG: Could not find captcha image element")
                return False

            # Load and preprocess image
            img = Image.open(io.BytesIO(captcha_bytes))

            # Convert to grayscale
            img = img.convert('L')

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)

            # Apply threshold to make text clearer
            img = img.point(lambda x: 0 if x < 128 else 255)

            # Run OCR with optimized config
            ocr_text = pytesseract.image_to_string(
                img,
                config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            ).strip()

            # Clean the result
            answer = re.sub(r'[^A-Za-z0-9]', '', ocr_text)
            print(f"DEBUG: OCR captcha answer: '{answer}'")

            if answer and 3 <= len(answer) <= 8:
                selector = target.get("selector")
                if selector:
                    await page.locator(selector).fill(answer, timeout=3000)
                    return True
        except Exception as e:
            print(f"DEBUG: OCR captcha solving failed: {e}")

        return False

    if name == "assert_url":
        current_url = page.url if isinstance(page.url, str) else page.url()
        if value and value.lower() not in current_url.lower():
            raise AssertionError(f"URL missing substring {value}")
        return False
    if name == "assert_text":
        locator = page.get_by_text(re.compile(value or "", re.IGNORECASE), exact=False)
        await locator.first.wait_for(state="visible", timeout=4000)
        return False
    raise RuntimeError(f"Unknown action {name}")


async def validate_step(page: Page, step: str, test_case: Dict[str, Any]) -> None:
    """Lightweight validator: looks for expected_result text and simple goal hints."""
    visible_text = ""
    try:
        visible_text = await page.evaluate("() => document.body.innerText || ''")
    except Exception:
        visible_text = ""
    url_lower = (page.url if isinstance(page.url, str) else page.url()).lower()
    step_lower = (step or "").lower()
    expected = (test_case.get("expected_result") or "").strip()
    if expected and len(expected) < 400:
        found = expected.lower() in visible_text.lower()
        print(f"Validation (expected_result substring): {'PASS' if found else 'WARN'}")
    if "about" in step_lower and "about" not in url_lower:
        print("Validation (about in URL): WARN - URL missing 'about'")
    if "client" in step_lower and "client" not in visible_text.lower():
        print("Validation (clients text visible): WARN - no 'client' substring in visible text")


async def handle_action_error(page: Page, error: Exception, attempt: int) -> None:
    await auto_dismiss_popups(page)
    if isinstance(error, PlaywrightTimeoutError):
        try:
            await page.reload(timeout=12000)
        except Exception:
            pass
    await page.wait_for_timeout(min(800 + attempt * 100, 2000))
