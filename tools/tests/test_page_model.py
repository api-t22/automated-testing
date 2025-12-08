print("STARTING TEST SCRIPT...")
import asyncio
import logging
import sys

from tools.agent_runner.page_model import fetch_and_build, human_processable

# Configure logging
logging.basicConfig(level=logging.INFO)


async def main():
    sys.stderr.write("--- Testing Page Model (The 'Eyes') ---\n")
    url = "https://www.cigroup.co.uk"
    sys.stderr.write(f"Fetching {url}...\n")

    # 1. Test fetch_and_build
    page_map = await fetch_and_build(url, headed=False)  # Headless for my environment

    # 2. Verify 'About Us' exists
    digest_items = page_map.get("digest", {}).get("nav", [])
    about_node = next((item for item in digest_items if "about" in (item.get("text") or "").lower()), None)

    if about_node:
        print(f"[PASS] Found 'About Us' node: {about_node.get('text')} (Ref: {about_node.get('ref')})")
    else:
        print("[FAIL] 'About Us' node NOT found in top nav.")

    # 3. Verify 'Clients' exists in subnav (expanded)
    subnav_items = page_map.get("digest", {}).get("subnav", [])
    client_node = next((item for item in subnav_items if "client" in (item.get("text") or "").lower()), None)

    if client_node:
        print(f"[PASS] Found 'Clients' subnav node: {client_node.get('text')} (Ref: {client_node.get('ref')})")
        # 4. Verify Structure (Parent Ref)
        if client_node.get("parentRef") is not None:
            print(f"[PASS] 'Clients' has parent_ref: {client_node.get('parentRef')}")
            # Check if it matches About
            if about_node and client_node.get("parentRef") == about_node.get("ref"):
                print("[PASS] Structure CORRECT: 'Clients' is a child of 'About Us'.")
            else:
                print(
                    f"[WARN] Structure mismatch? About Ref={about_node.get('ref') if about_node else 'None'}, Client Parent={client_node.get('parentRef')}")
        else:
            print("[FAIL] 'Clients' node has NO parent_ref (Structure missing).")
    else:
        print("[FAIL] 'Clients' subnav node NOT found. (Expansion failed?)")

    print("\n--- Human Readable Digest ---")
    print(human_processable(page_map))


if __name__ == "__main__":
    asyncio.run(main())
