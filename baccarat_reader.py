import os
import re
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Browser, Page, Playwright, async_playwright

from color_reader import parse_road_from_screenshot
from config import (
    DEFAULT_TABLE_IDS,
    HALL_LABELS,
    HEADLESS,
    USE_COLOR_READER,
    USE_DOM_READER,
    get_platform,
)


class BaccaratReader:
    """
    通用百家樂頁面讀取器。
    它不繞過登入、不破解驗證，只讀取你提供網址打開後頁面上可見的資料。
    """

    def __init__(self) -> None:
        self.viewport = {"width": 1280, "height": 900}

    async def _open_page(self, platform_key: str) -> Tuple[Playwright, Browser, Page]:
        platform = get_platform(platform_key)
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=HEADLESS,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = await browser.new_context(
            viewport=self.viewport,
            ignore_https_errors=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        await page.goto(platform.url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)
        return playwright, browser, page

    async def _close(self, playwright: Optional[Playwright], browser: Optional[Browser]) -> None:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
        if playwright:
            try:
                await playwright.stop()
            except Exception:
                pass

    async def _click_any_text(self, page: Page, labels: List[str]) -> bool:
        for label in labels:
            if not label:
                continue
            for frame in page.frames:
                try:
                    locator = frame.get_by_text(label, exact=False)
                    if await locator.count() > 0:
                        await locator.first.click(timeout=3000)
                        await page.wait_for_timeout(2000)
                        return True
                except Exception:
                    continue
        return False

    async def _extract_visible_text(self, page: Page) -> str:
        parts: List[str] = []
        for frame in page.frames:
            try:
                text = await frame.locator("body").inner_text(timeout=3000)
                if text:
                    parts.append(text)
            except Exception:
                continue
        return "\n".join(dict.fromkeys(parts))

    async def _select_hall(self, page: Page, hall_key: str) -> None:
        labels = HALL_LABELS.get(hall_key, [])
        if labels:
            await self._click_any_text(page, labels)

    async def _select_table(self, page: Page, table_id: str) -> None:
        labels = [table_id, table_id.upper(), table_id.lower()]
        await self._click_any_text(page, labels)

    def _parse_table_list_from_text(self, text: str) -> List[Dict[str, Any]]:
        table_ids: List[str] = []
        patterns = [r"\bRB\d{1,2}\b", r"\bSB\d{1,2}\b", r"\bCB\d{1,2}\b", r"\bR\d{1,2}\b"]

        for pattern in patterns:
            table_ids.extend(re.findall(pattern, text, flags=re.IGNORECASE))

        table_ids = list(dict.fromkeys([x.upper() for x in table_ids]))
        game_nos = re.findall(r"20\d{8,}[A-Z0-9]*", text, flags=re.IGNORECASE)
        dealers = re.findall(
            r"(?:荷官姓名|荷官|Dealer)\s*[:：]?\s*([A-Za-z\u4e00-\u9fff0-9_-]{1,20})",
            text,
            flags=re.IGNORECASE,
        )
        online_counts = re.findall(
            r"(?:在線人數|在线人数|Online)\s*[:：]?\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )

        max_len = max(len(table_ids), len(game_nos), len(dealers), len(online_counts), 0)
        tables: List[Dict[str, Any]] = []

        for i in range(max_len):
            table_id = table_ids[i] if i < len(table_ids) else ""
            if not table_id and i < len(DEFAULT_TABLE_IDS):
                table_id = DEFAULT_TABLE_IDS[i]

            if not table_id:
                continue

            tables.append(
                {
                    "table_id": table_id,
                    "game_no": game_nos[i] if i < len(game_nos) else "",
                    "dealer": dealers[i] if i < len(dealers) else "",
                    "online_count": int(online_counts[i]) if i < len(online_counts) and online_counts[i].isdigit() else 0,
                }
            )

        if not tables:
            tables = [{"table_id": table_id, "game_no": "", "dealer": "", "online_count": 0} for table_id in DEFAULT_TABLE_IDS]

        return tables[:12]

    async def _extract_dom_road(self, page: Page) -> List[str]:
        if not USE_DOM_READER:
            return []

        script = """
        () => {
            const out = [];
            const nodes = Array.from(document.querySelectorAll('div,span,i,li,td,p,circle'));

            for (const n of nodes) {
                const text = (n.innerText || n.textContent || '').trim();
                const cls = typeof n.className === 'string' ? n.className : '';
                const style = n.getAttribute('style') || '';
                const aria = n.getAttribute('aria-label') || '';
                const title = n.getAttribute('title') || '';
                const data = [
                    n.getAttribute('data-result') || '',
                    n.getAttribute('data-type') || '',
                    n.getAttribute('data-value') || '',
                    n.getAttribute('data-side') || '',
                ].join(' ');

                const combo = `${text} ${cls} ${style} ${aria} ${title} ${data}`.toLowerCase();

                const hasRoadWord =
                    combo.includes('road') || combo.includes('result') || combo.includes('bead') ||
                    combo.includes('banker') || combo.includes('player') || combo.includes('tie') ||
                    combo.includes('莊') || combo.includes('庄') || combo.includes('閒') || combo.includes('闲') || combo.includes('和');

                if (!hasRoadWord) continue;

                if (combo.includes('banker') || combo.includes('莊') || combo.includes('庄') || combo === 'b') {
                    out.push('B');
                } else if (combo.includes('player') || combo.includes('閒') || combo.includes('闲') || combo === 'p') {
                    out.push('P');
                } else if (combo.includes('tie') || combo.includes('和') || combo === 't') {
                    out.push('T');
                }
            }
            return out.slice(-120);
        }
        """

        results: List[str] = []
        for frame in page.frames:
            try:
                values = await frame.evaluate(script)
                if values:
                    results.extend([x for x in values if x in ("B", "P", "T")])
            except Exception:
                continue
        return results[-120:]

    async def _extract_color_road(self, page: Page, platform_key: str, table_id: str) -> List[str]:
        if not USE_COLOR_READER:
            return []

        os.makedirs("tmp", exist_ok=True)
        path = f"tmp/{platform_key}_{table_id}.png"
        try:
            await page.screenshot(path=path, full_page=True)
            return parse_road_from_screenshot(path)
        except Exception:
            return []

    def _parse_basic_data(self, text: str, fallback_table_id: str, road: List[str]) -> Dict[str, Any]:
        table_id = fallback_table_id

        table_match = re.search(r"(?:桌號|桌号|Table)\s*[:：]?\s*([A-Za-z0-9_-]{2,20})", text, flags=re.IGNORECASE)
        if table_match:
            table_id = table_match.group(1).upper()

        game_no = ""
        game_match = re.search(r"20\d{8,}[A-Z0-9]*", text, flags=re.IGNORECASE)
        if game_match:
            game_no = game_match.group(0)

        countdown = 0
        countdown_match = re.search(r"(?:倒數計時|倒数计时|倒數|倒数|countdown)\s*[:：]?\s*(\d+)", text, flags=re.IGNORECASE)
        if countdown_match:
            countdown = int(countdown_match.group(1))

        dealer = ""
        dealer_match = re.search(r"(?:荷官姓名|荷官|Dealer)\s*[:：]?\s*([A-Za-z\u4e00-\u9fff0-9_-]{1,20})", text, flags=re.IGNORECASE)
        if dealer_match:
            dealer = dealer_match.group(1)

        online_count = 0
        online_match = re.search(r"(?:在線人數|在线人数|Online)\s*[:：]?\s*(\d+)", text, flags=re.IGNORECASE)
        if online_match:
            online_count = int(online_match.group(1))

        lower = text.lower()
        if "可押注" in text or "可下注" in text or "bet" in lower:
            status = "可押注"
        elif "停止下注" in text or "不可押注" in text:
            status = "停止下注"
        else:
            status = "讀取中"

        last_result = road[-1] if road else ""
        round_key = f"{table_id}:{game_no}:{len(road)}:{last_result}:{countdown}:{status}"

        return {
            "table_id": table_id,
            "game_no": game_no,
            "dealer": dealer,
            "online_count": online_count,
            "countdown": countdown,
            "status": status,
            "road": road,
            "round_key": round_key,
        }

    async def list_tables(self, platform_key: str, hall_key: str) -> List[Dict[str, Any]]:
        playwright: Optional[Playwright] = None
        browser: Optional[Browser] = None
        try:
            playwright, browser, page = await self._open_page(platform_key)
            await self._select_hall(page, hall_key)
            text = await self._extract_visible_text(page)
            return self._parse_table_list_from_text(text)
        finally:
            await self._close(playwright, browser)

    async def read_table_data(self, platform_key: str, hall_key: str, table_id: str) -> Dict[str, Any]:
        playwright: Optional[Playwright] = None
        browser: Optional[Browser] = None
        try:
            playwright, browser, page = await self._open_page(platform_key)
            await self._select_hall(page, hall_key)
            await self._select_table(page, table_id)
            await page.wait_for_timeout(2500)

            text = await self._extract_visible_text(page)
            road = await self._extract_dom_road(page)

            if len(road) < 3:
                color_road = await self._extract_color_road(page, platform_key, table_id)
                if color_road:
                    road = color_road

            return self._parse_basic_data(text=text, fallback_table_id=table_id, road=road)
        finally:
            await self._close(playwright, browser)
