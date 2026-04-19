"""Capture & validate real screenshots of the Streamlit UI for thesis.

Runs against a live Streamlit server (default http://127.0.0.1:8501).
Captures 10 screens, validates each PNG (size, dimensions, non-blank content)
and fails loudly if any screenshot looks broken.

Usage:
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from playwright.async_api import Page, async_playwright

STREAMLIT_URL = os.getenv("STREAMLIT_URL", "http://127.0.0.1:8501")
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Validation thresholds
MIN_FILE_KB = 40       # An empty page is usually ~10 KB
MIN_WIDTH = 1200
MIN_HEIGHT = 700
MIN_NONWHITE_RATIO = 0.05   # At least 5% non-white pixels — else probably blank


def validate_png(path: Path, description: str) -> tuple[bool, str]:
    """Check that the screenshot is non-trivial."""
    if not path.exists():
        return False, f"missing file"
    size_kb = path.stat().st_size / 1024
    if size_kb < MIN_FILE_KB:
        return False, f"file too small: {size_kb:.0f} KB < {MIN_FILE_KB}"

    # Check dimensions + content using PIL
    try:
        from PIL import Image
        img = Image.open(path)
        w, h = img.size
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return False, f"dimensions {w}x{h} below minimum {MIN_WIDTH}x{MIN_HEIGHT}"
        # Sample pixels for non-white content
        img_rgb = img.convert("RGB").resize((120, 80))
        import numpy as np
        arr = np.array(img_rgb)
        non_white = ((arr < 250).any(axis=2)).sum()
        total = arr.shape[0] * arr.shape[1]
        ratio = non_white / total
        if ratio < MIN_NONWHITE_RATIO:
            return False, f"mostly blank ({ratio*100:.1f}% non-white)"
        return True, f"OK  {size_kb:.0f} KB  {w}x{h}  {ratio*100:.0f}% content"
    except Exception as e:
        return False, f"PIL error: {e}"


async def capture_and_validate(page: Page, filename: str, description: str) -> bool:
    path = OUTPUT_DIR / filename
    await page.screenshot(path=str(path), full_page=True)
    ok, msg = validate_png(path, description)
    status = "[OK]" if ok else "[XX]"
    print(f"  {status} {filename:45s} | {description:40s} | {msg}")
    return ok


async def login(page: Page, username: str = "planner", password: str = "planner123"):
    await page.goto(STREAMLIT_URL, wait_until="networkidle")
    await page.wait_for_timeout(3500)
    text_inputs = page.locator("input[type='text']")
    password_inputs = page.locator("input[type='password']")
    await text_inputs.first.wait_for(state="visible", timeout=20000)
    await text_inputs.first.fill(username)
    await password_inputs.first.fill(password)
    submit = page.locator("button[kind='primaryFormSubmit'], "
                          "button[data-testid='stBaseButton-primaryFormSubmit']").first
    await submit.click()
    await page.wait_for_timeout(3000)
    await wait_for_streamlit_idle(page, max_wait_ms=30000)


async def wait_for_streamlit_idle(page: Page, max_wait_ms: int = 30000):
    """Wait until Streamlit finishes rendering (Running indicator gone).

    Streamlit adds a 'Running...' status near the top-right while executing
    the script. When idle, the button disappears / changes.
    """
    elapsed = 0
    step = 500
    while elapsed < max_wait_ms:
        # The "Stop" / "Running" indicator has testid stStatusWidget when active.
        running = await page.locator("[data-testid='stStatusWidget']").count()
        if running == 0:
            # Extra settle time for final renders (plotly, images)
            await page.wait_for_timeout(1500)
            return True
        await page.wait_for_timeout(step)
        elapsed += step
    print(f"  [!] Streamlit did not reach idle within {max_wait_ms}ms")
    return False


async def navigate_to_page(page: Page, partial_label: str) -> bool:
    await page.wait_for_timeout(1200)
    links = page.locator("[data-testid='stSidebarNav'] a")
    count = await links.count()
    for i in range(count):
        text = (await links.nth(i).text_content() or "").strip()
        if partial_label.lower() in text.lower():
            await links.nth(i).click()
            # Wait initial response then wait for idle
            await page.wait_for_timeout(1500)
            await wait_for_streamlit_idle(page, max_wait_ms=45000)
            return True
    print(f"  [!] Page '{partial_label}' not found in sidebar")
    return False


async def run_capture() -> dict:
    """Returns a dict of {filename: ok_bool}."""
    results: dict[str, bool] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1600, "height": 1000},
                                              device_scale_factor=1)
        page = await context.new_page()

        print("\n=== Session 1: planner login ===")
        await page.goto(STREAMLIT_URL, wait_until="networkidle")
        await page.wait_for_timeout(3000)
        results["ui_01_login.png"] = await capture_and_validate(
            page, "ui_01_login.png", "Login screen")

        await login(page)
        results["ui_02_dashboard_main.png"] = await capture_and_validate(
            page, "ui_02_dashboard_main.png", "Landing after login")

        if await navigate_to_page(page, "Dashboard"):
            results["ui_03_dashboard.png"] = await capture_and_validate(
                page, "ui_03_dashboard.png", "Dashboard page")

        if await navigate_to_page(page, "Route Analysis"):
            await page.wait_for_timeout(2000)
            results["ui_04_route_analysis.png"] = await capture_and_validate(
                page, "ui_04_route_analysis.png", "Route Analysis (before forecast)")

            # Click forecast
            try:
                btn = page.get_by_role("button", name="Построить прогноз")
                await btn.click()
                await page.wait_for_timeout(18000)
                results["ui_05_route_analysis_forecast.png"] = await capture_and_validate(
                    page, "ui_05_route_analysis_forecast.png",
                    "Route Analysis with forecast")
            except Exception as e:
                print(f"  [!] Forecast button error: {e}")
                results["ui_05_route_analysis_forecast.png"] = False

        if await navigate_to_page(page, "Models Comparison"):
            await page.wait_for_timeout(2000)
            try:
                btn = page.get_by_role("button", name="Запустить сравнение")
                await btn.click()
                await page.wait_for_timeout(22000)
            except Exception as e:
                print(f"  [!] Compare button error: {e}")
            results["ui_06_models_comparison.png"] = await capture_and_validate(
                page, "ui_06_models_comparison.png", "Models comparison")

        if await navigate_to_page(page, "Monitoring"):
            results["ui_07_monitoring.png"] = await capture_and_validate(
                page, "ui_07_monitoring.png", "Monitoring page")

        if await navigate_to_page(page, "Reports"):
            results["ui_08_reports.png"] = await capture_and_validate(
                page, "ui_08_reports.png", "Reports page")

        await context.close()

        print("\n=== Session 2: admin login (for Model Management + Admin) ===")
        context2 = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page2 = await context2.new_page()
        await login(page2, "admin", "admin123")

        if await navigate_to_page(page2, "Model Management"):
            results["ui_09_model_management.png"] = await capture_and_validate(
                page2, "ui_09_model_management.png", "Model Management (admin)")

        if await navigate_to_page(page2, "Admin"):
            results["ui_10_admin.png"] = await capture_and_validate(
                page2, "ui_10_admin.png", "Admin page")

        await context2.close()
        await browser.close()

    return results


def main():
    print(f"Target: {STREAMLIT_URL}")
    print(f"Output: {OUTPUT_DIR}")

    results = asyncio.run(run_capture())

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    ok_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{ok_count}/{total} screenshots passed validation")
    if ok_count < total:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {failed}")
        sys.exit(1)
    print("All screenshots valid [OK]")


if __name__ == "__main__":
    main()
