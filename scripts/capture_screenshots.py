"""Capture real screenshots of the Streamlit UI for thesis figures.

Assumes Streamlit is running on http://127.0.0.1:8501 and FastAPI on :8000.
"""

import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

STREAMLIT_URL = os.getenv("STREAMLIT_URL", "http://127.0.0.1:8501")
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def login(page, username: str = "planner", password: str = "planner123"):
    """Fill login form using input type selectors (more reliable than labels)."""
    await page.goto(STREAMLIT_URL, wait_until="networkidle")
    await page.wait_for_timeout(3000)

    # Streamlit text_input renders as <input type="text"> within a container
    text_inputs = page.locator("input[type='text']")
    password_inputs = page.locator("input[type='password']")

    # Wait for inputs to be available
    await text_inputs.first.wait_for(state="visible", timeout=15000)

    await text_inputs.first.fill(username)
    await password_inputs.first.fill(password)

    # Click "Войти" — submit button
    submit_button = page.locator("button[kind='primaryFormSubmit'], button[data-testid='stBaseButton-primaryFormSubmit']").first
    await submit_button.click()
    await page.wait_for_timeout(10000)


async def login_as(page, username: str, password: str):
    await login(page, username, password)


async def navigate_to_page(page, partial_label: str):
    """Click a page link in the Streamlit sidebar nav."""
    await page.wait_for_timeout(1500)
    # Streamlit page links are <a> with class st-... in sidebar
    links = page.locator("[data-testid='stSidebarNav'] a")
    count = await links.count()
    for i in range(count):
        text = (await links.nth(i).text_content() or "").strip()
        if partial_label.lower() in text.lower():
            await links.nth(i).click()
            await page.wait_for_timeout(4000)
            return True
    print(f"[!] Page with label '{partial_label}' not found")
    return False


async def capture(page, filename: str):
    path = OUTPUT_DIR / filename
    await page.screenshot(path=str(path), full_page=True)
    print(f"Saved: {path}")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1600, "height": 1000},
                                              device_scale_factor=1)
        page = await context.new_page()

        # 1. Login page (unauthenticated)
        await page.goto(STREAMLIT_URL, wait_until="networkidle")
        await page.wait_for_timeout(3000)
        await capture(page, "ui_01_login.png")

        # Log in
        await login(page)
        await capture(page, "ui_02_dashboard_main.png")

        # 2. Dashboard page
        if await navigate_to_page(page, "Dashboard"):
            await capture(page, "ui_03_dashboard.png")

        # 3. Route Analysis
        if await navigate_to_page(page, "Route Analysis"):
            await page.wait_for_timeout(2000)
            await capture(page, "ui_04_route_analysis.png")

            # Click forecast button
            try:
                await page.get_by_role("button", name="Построить прогноз").click()
                await page.wait_for_timeout(15000)  # Model training
                await capture(page, "ui_05_route_analysis_forecast.png")
            except Exception as e:
                print(f"Forecast button error: {e}")

        # 4. Models Comparison
        if await navigate_to_page(page, "Models Comparison"):
            await page.wait_for_timeout(2000)
            try:
                await page.get_by_role("button", name="Запустить сравнение").click()
                await page.wait_for_timeout(20000)
            except Exception:
                pass
            await capture(page, "ui_06_models_comparison.png")

        # 5. Monitoring
        if await navigate_to_page(page, "Monitoring"):
            await capture(page, "ui_07_monitoring.png")

        # 6. Reports
        if await navigate_to_page(page, "Reports"):
            await capture(page, "ui_08_reports.png")

        await context.close()

        # 7. Login as admin for Model Management + Admin pages
        context2 = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page2 = await context2.new_page()
        await login_as(page2, "admin", "admin123")

        if await navigate_to_page(page2, "Model Management"):
            await capture(page2, "ui_09_model_management.png")

        if await navigate_to_page(page2, "Admin"):
            await capture(page2, "ui_10_admin.png")

        await context2.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
