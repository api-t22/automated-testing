import asyncio
import logging
import sys

from playwright.async_api import async_playwright

from tools.agent_runner.actions import apply_action
from tools.agent_runner.page_model import collect_clickable_elements, expand_nav_and_collect

# Configure logging
logging.basicConfig(level=logging.INFO)


async def main():
    sys.stderr.write("--- Testing Actions (The 'Hands') ---\n")
    url = "https://www.cigroup.co.uk"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Check headless but assume it works
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        sys.stderr.write(f"Navigating to {url}...\n")
        await page.goto(url, wait_until="networkidle", timeout=30000)

        # 1. Collect elements
        print("Collecting elements...")
        base = await collect_clickable_elements(page)
        expanded = await expand_nav_and_collect(page, base)

        # 2. Find Accept Button
        accept_btn = next((el for el in expanded if "accept" in (el.get("text") or "").lower()), None)
        if accept_btn:
            print(f"Found Accept Button: {accept_btn.get('text')} (Ref {accept_btn.get('index')})")
            # Action: Click Accept
            action = {"action": "click", "primary_ref": accept_btn.get("index"), "value": "Accept"}
            print(f"Applying action: {action}")
            await apply_action(page, action, expanded)
            print("Action applied. Waiting...")
            await asyncio.sleep(2)
        else:
            print("Accept button not found (Maybe already gone?).")

        # 3. Find About Us
        about_btn = next((el for el in expanded if "about" in (el.get("text") or "").lower()), None)
        if about_btn:
            print(f"Found About Us: {about_btn.get('text')} (Ref {about_btn.get('index')})")
            # Action: Hover Click About -> Clients
            # Find Clients child
            clients_btn = next((el for el in expanded if
                                "client" in (el.get("text") or "").lower() and el.get("parentRef") == about_btn.get(
                                    "index")), None)

            if clients_btn:
                print(f"Found Clients Subnav: {clients_btn.get('text')} (Ref {clients_btn.get('index')})")
                action = {
                    "action": "hover_click",
                    "primary_ref": about_btn.get("index"),
                    "secondary_ref": clients_btn.get("index"),
                    "value": "Clients"
                }
                print(f"Applying complex action: {action}")
                await apply_action(page, action, expanded)
                print("Action applied. Checking URL...")
                await asyncio.sleep(2)
                print(f"Current URL: {page.url}")
                if "client" in page.url.lower():
                    print("[PASS] Successfully navigated to Clients page!")
                else:
                    print("[FAIL] URL did not change to Clients.")
            else:
                print("[FAIL] Could not find 'Clients' child of 'About Us'. check expansion.")
        else:
            print("[FAIL] Could not find 'About Us'.")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
