import asyncio
import os
# Ensure PYTHONPATH includes current directory
import sys

from tools.agent_runner.run import run_test_plan

sys.path.append(os.getcwd())


async def main():
    # Define test plan as a Python dictionary
    my_plan = {
        "test_cases": [
            {
                "id": "TC-DEMO-001",
                "title": "Verify Login",
                "steps": [
                    "Type 'standard_user' into username",
                    "Type 'secret_sauce' into password",
                    "Click Login"
                ],
                "test_data": {
                    "expected_url": "/inventory"
                }
            }
        ]
    }

    print("Running test plan programmatically...")

    # Call the API
    results = await run_test_plan(
        url="https://www.saucedemo.com",
        test_plan=my_plan,
        headed=True,  # Visible browser
        persist_session=False
    )

    print("\nAPI Results:")
    for r in results:
        print(f"{r['id']}: {r['status']}")


if __name__ == "__main__":
    asyncio.run(main())
