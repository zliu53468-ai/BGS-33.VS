# baccarat_reader.py
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, Browser, Playwright

from config import (
    get_platform,
    HEADLESS,
    USE_DOM_READER,
    USE_NETWORK_READER,
    USE_COLOR_READER,
    ALLOW_DEFAULT_TABLE_IDS,
    DEFAULT_TABLE_IDS,
    READER_WAIT_MS,
    READER_TIMEOUT_MS,
    BROWSER_WIDTH,
    BROWSER_HEIGHT,
)
from color_reader import parse_road_from_screenshot


BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-blink-features=AutomationControlled",
]

TABLE_ID_PATTERNS = [
    r"\bRB\d{1,3}\b",
    r"\bSB\d{1,3}\b",
    r"\bCB\d{1,3}\b",
    r"\bTB\d{1,3}\b",
    r"\bBAC\d{1,4}\b",
    r"\bB\d{1,4}\b",
    r"\bR\d{1,5}\b",
    r"\b[A-Z]{1,4}-?\d{1,4}\b",
    r"\d{1,3}\s*號桌",
    r"百家樂\s*\d{1,3}",
    r"百家乐\s*\d{1,3}",
]

GAME_NO_PATTERNS = [
    r"20\d{8,}[A-Z0-9_-]*",
    r"[A-Z]{1,4}\d{8,}[A-Z0-9_-]*",
]

RESULT_MAP = {
    "B": "B", "BANKER": "B", "莊": "B", "庄": "B", "BANK": "B",
    "P": "P", "PLAYER": "P", "閒": "P", "闲": "P", "PLAY": "P",
    "T": "T", "TIE": "T", "和": "T",
}


def _uniq(seq: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in seq:
        key = str(x).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _safe_json_text(value: Any, limit: int = 1200) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)[:limit]
    except Exception:
        return str(value)[:limit]


def _normalize_result(value: Any) -> str:
    if value is None:
        return ""
    v = str(value).strip().upper()
    if v in RESULT_MAP:
        return RESULT_MAP[v]
    # 部分 API 用 1/2/3 或 0/1/2 表示，這裡不強猜，避免誤判。
    return ""


def _walk_json(obj: Any, results: Dict[str, Any]) -> None:
    """從 API JSON 中盡量挖桌台、牌路、局號資料。"""
    if isinstance(obj, dict):
        lower_keys = {str(k).lower(): k for k in obj.keys()}

        # 桌台資料
        table_id = None
        for key in ("tableid", "table_id", "tableno", "table_no", "table", "vid", "roomid", "room_id"):
            if key in lower_keys:
                table_id = str(obj.get(lower_keys[key], "")).strip()
                break
        if table_id:
            item = {
                "table_id": table_id.upper(),
                "game_no": "",
                "dealer": "",
                "online_count": 0,
                "source": "network_json",
            }
            for key in ("gameno", "game_no", "roundno", "round_no", "shoe", "boot"):
                if key in lower_keys:
                    item["game_no"] = str(obj.get(lower_keys[key], "")).strip()
                    break
            for key in ("dealer", "dealername", "dealer_name", "荷官", "荷官姓名"):
                if key in lower_keys:
                    item["dealer"] = str(obj.get(lower_keys[key], "")).strip()
                    break
            for key in ("online", "onlinecount", "online_count", "players", "playercount", "usercount"):
                if key in lower_keys:
                    try:
                        item["online_count"] = int(float(obj.get(lower_keys[key], 0) or 0))
                    except Exception:
                        pass
                    break
            results.setdefault("tables", []).append(item)

        # 牌路資料
        for key in ("road", "roads", "beadroad", "bigroad", "results", "history", "rounds", "resultlist", "result_list"):
            if key in lower_keys:
                road = _extract_road_from_any(obj.get(lower_keys[key]))
                if road:
                    results.setdefault("roads", []).append(road)

        # 單一結果
        for key in ("result", "winner", "win", "side", "outcome"):
            if key in lower_keys:
                r = _normalize_result(obj.get(lower_keys[key]))
                if r:
                    results.setdefault("single_results", []).append(r)

        for v in obj.values():
            _walk_json(v, results)
    elif isinstance(obj, list):
        road = _extract_road_from_any(obj)
        if len(road) >= 3:
            results.setdefault("roads", []).append(road)
        for v in obj:
            _walk_json(v, results)


def _extract_road_from_any(value: Any) -> List[str]:
    road: List[str] = []
    if isinstance(value, str):
        # 只接受看起來像連續牌路的字串，避免一般文字誤判。
        compact = re.sub(r"[^BPT莊庄閒闲和]", "", value.upper())
        if len(compact) >= 3:
            for ch in compact:
                r = _normalize_result(ch)
                if r:
                    road.append(r)
        return road
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                found = ""
                for key in ("result", "winner", "win", "side", "outcome", "value"):
                    if key in item:
                        found = _normalize_result(item.get(key))
                        if found:
                            break
                if found:
                    road.append(found)
            else:
                r = _normalize_result(item)
                if r:
                    road.append(r)
        return road
    if isinstance(value, dict):
        return _extract_road_from_any(list(value.values()))
    return []


class NetworkCollector:
    def __init__(self):
        self.json_payloads: List[Any] = []
        self.text_payloads: List[str] = []
        self.urls: List[str] = []

    def attach(self, page: Page) -> None:
        if not USE_NETWORK_READER:
            return

        async def on_response(response):
            try:
                url = response.url
                ctype = (response.headers or {}).get("content-type", "").lower()
                if any(x in url.lower() for x in (".png", ".jpg", ".jpeg", ".gif", ".woff", ".css")):
                    return
                if len(self.urls) < 120:
                    self.urls.append(url)
                if "json" in ctype or "api" in url.lower() or "game" in url.lower() or "table" in url.lower():
                    try:
                        data = await response.json()
                        self.json_payloads.append(data)
                    except Exception:
                        try:
                            text = await response.text()
                            if text and len(text) < 300000:
                                self.text_payloads.append(text[:5000])
                        except Exception:
                            pass
            except Exception:
                pass

        page.on("response", lambda response: asyncio.create_task(on_response(response)))

    def extract(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"tables": [], "roads": [], "single_results": []}
        for data in self.json_payloads[-80:]:
            _walk_json(data, results)
        for text in self.text_payloads[-30:]:
            # 嘗試從文字裡找 JSON
            for m in re.finditer(r"\{.*?\}|\[.*?\]", text, flags=re.S):
                snippet = m.group(0)
                if len(snippet) > 20000:
                    continue
                try:
                    obj = json.loads(snippet)
                    _walk_json(obj, results)
                except Exception:
                    continue
        return results


class BaccaratReader:
    def __init__(self):
        self.viewport = {"width": BROWSER_WIDTH, "height": BROWSER_HEIGHT}

    async def _new_page(self, platform_key: str) -> Tuple[Playwright, Browser, Page, NetworkCollector]:
        platform = get_platform(platform_key)
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=HEADLESS, args=BROWSER_ARGS)
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
        collector = NetworkCollector()
        collector.attach(page)
        await page.goto(platform.url, wait_until="domcontentloaded", timeout=READER_TIMEOUT_MS)
        try:
            await page.wait_for_load_state("networkidle", timeout=READER_TIMEOUT_MS)
        except Exception:
            pass
        await page.wait_for_timeout(READER_WAIT_MS)
        return playwright, browser, page, collector

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
                        await locator.first.click(timeout=4000)
                        await page.wait_for_timeout(2500)
                        return True
                except Exception:
                    continue
            # JS 文字模糊點擊備用
            try:
                ok = await page.evaluate(
                    """(label) => {
                        const nodes = Array.from(document.querySelectorAll('button,a,div,span,li'));
                        const n = nodes.find(x => (x.innerText || x.textContent || '').includes(label));
                        if (n) { n.click(); return true; }
                        return false;
                    }""",
                    label,
                )
                if ok:
                    await page.wait_for_timeout(2500)
                    return True
            except Exception:
                pass
        return False

    async def _extract_dom_dump(self, page: Page) -> Dict[str, Any]:
        texts: List[str] = []
        attrs: List[str] = []
        url_list: List[str] = []
        frame_count = len(page.frames)

        script = """
        () => {
          const data = { text: '', attrs: [], hrefs: [] };
          data.text = document.body ? (document.body.innerText || document.body.textContent || '') : '';
          const nodes = Array.from(document.querySelectorAll('*')).slice(0, 6000);
          for (const n of nodes) {
            const parts = [];
            for (const a of ['id','class','title','aria-label','alt','data-table','data-table-id','data-id','data-room','data-game','data-result','data-type','data-value','data-side']) {
              const v = n.getAttribute && n.getAttribute(a);
              if (v) parts.push(`${a}=${v}`);
            }
            const t = (n.innerText || n.textContent || '').trim();
            if (t && t.length <= 80) parts.push(`text=${t}`);
            if (parts.length) data.attrs.push(parts.join(' | '));
            const href = n.getAttribute && (n.getAttribute('href') || n.getAttribute('src'));
            if (href) data.hrefs.push(href);
          }
          return data;
        }
        """
        for frame in page.frames:
            try:
                dump = await frame.evaluate(script)
                if dump.get("text"):
                    texts.append(dump["text"])
                attrs.extend(dump.get("attrs") or [])
                url_list.extend(dump.get("hrefs") or [])
            except Exception:
                continue
        return {
            "text": "\n".join(_uniq(texts)),
            "attrs": _uniq(attrs)[:2500],
            "hrefs": _uniq(url_list)[:500],
            "frame_count": frame_count,
        }

    def _parse_tables_from_blob(self, blob: str, source: str = "dom_text") -> List[Dict[str, Any]]:
        tables: Dict[str, Dict[str, Any]] = {}
        if not blob:
            return []

        for pattern in TABLE_ID_PATTERNS:
            for m in re.finditer(pattern, blob, flags=re.IGNORECASE):
                table_id = re.sub(r"\s+", "", m.group(0)).upper()
                # 避免把超長局號切成桌號誤收
                if len(table_id) > 18:
                    continue
                if table_id not in tables:
                    tables[table_id] = {
                        "table_id": table_id,
                        "game_no": "",
                        "dealer": "",
                        "online_count": 0,
                        "source": source,
                    }

        # 嘗試找局號、荷官、在線人數，但不硬配對；只補到第一個資料。
        game_nos: List[str] = []
        for p in GAME_NO_PATTERNS:
            game_nos.extend(re.findall(p, blob, flags=re.IGNORECASE))
        game_nos = _uniq([x.strip() for x in game_nos])

        dealers = re.findall(
            r"(?:荷官姓名|荷官|Dealer|dealerName|dealer_name)\s*[:：=]?\s*([A-Za-z\u4e00-\u9fff0-9_-]{1,30})",
            blob,
            flags=re.IGNORECASE,
        )
        online_counts = re.findall(
            r"(?:在線人數|在线人数|Online|onlineCount|players|userCount)\s*[:：=]?\s*(\d+)",
            blob,
            flags=re.IGNORECASE,
        )

        table_items = list(tables.values())
        for i, item in enumerate(table_items):
            if i < len(game_nos):
                item["game_no"] = game_nos[i]
            if i < len(dealers):
                item["dealer"] = dealers[i]
            if i < len(online_counts):
                try:
                    item["online_count"] = int(online_counts[i])
                except Exception:
                    pass

        return table_items

    def _merge_tables(self, *groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            for item in group or []:
                table_id = str(item.get("table_id", "")).strip().upper()
                if not table_id:
                    continue
                if table_id not in merged:
                    merged[table_id] = {
                        "table_id": table_id,
                        "game_no": item.get("game_no", "") or "",
                        "dealer": item.get("dealer", "") or "",
                        "online_count": item.get("online_count", 0) or 0,
                        "source": item.get("source", "unknown"),
                    }
                else:
                    if item.get("game_no") and not merged[table_id].get("game_no"):
                        merged[table_id]["game_no"] = item.get("game_no")
                    if item.get("dealer") and not merged[table_id].get("dealer"):
                        merged[table_id]["dealer"] = item.get("dealer")
                    if item.get("online_count") and not merged[table_id].get("online_count"):
                        merged[table_id]["online_count"] = item.get("online_count")
                    merged[table_id]["source"] = merged[table_id].get("source", "") + "+" + str(item.get("source", ""))
        return list(merged.values())[:50]

    async def _extract_dom_road(self, page: Page) -> List[str]:
        if not USE_DOM_READER:
            return []
        script = """
        () => {
          const out = [];
          const nodes = Array.from(document.querySelectorAll('div,span,i,li,td,p,button')).slice(0, 8000);
          for (const n of nodes) {
            const rect = n.getBoundingClientRect();
            if (!rect || rect.width < 2 || rect.height < 2) continue;
            const text = (n.innerText || n.textContent || '').trim();
            const cls = typeof n.className === 'string' ? n.className : '';
            const style = n.getAttribute('style') || '';
            const data = ['data-result','data-type','data-value','data-side','aria-label','title']
              .map(a => n.getAttribute(a) || '').join(' ');
            const combo = `${text} ${cls} ${style} ${data}`.toLowerCase();
            const looks = combo.includes('road') || combo.includes('result') || combo.includes('banker') || combo.includes('player') || combo.includes('tie') || combo.includes('莊') || combo.includes('庄') || combo.includes('閒') || combo.includes('闲') || combo.includes('和');
            if (!looks) continue;
            if (combo.includes('banker') || combo.includes('莊') || combo.includes('庄') || combo === 'b') out.push('B');
            else if (combo.includes('player') || combo.includes('閒') || combo.includes('闲') || combo === 'p') out.push('P');
            else if (combo.includes('tie') || combo.includes('和') || combo === 't') out.push('T');
          }
          return out.slice(-160);
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
        return results[-160:]

    async def _extract_color_road(self, page: Page, platform_key: str, table_id: str) -> List[str]:
        if not USE_COLOR_READER:
            return []
        os.makedirs("tmp", exist_ok=True)
        screenshot_path = f"tmp/{platform_key}_{table_id}.png"
        try:
            await page.screenshot(path=screenshot_path, full_page=True)
            return parse_road_from_screenshot(screenshot_path)
        except Exception:
            return []

    def _parse_basic_data(self, blob: str, fallback_table_id: str, road: List[str], network: Dict[str, Any]) -> Dict[str, Any]:
        table_id = fallback_table_id
        game_no = ""
        dealer = ""
        online_count = 0
        countdown = 0

        game_match = re.search(r"20\d{8,}[A-Z0-9_-]*", blob, flags=re.IGNORECASE)
        if game_match:
            game_no = game_match.group(0)

        dealer_match = re.search(
            r"(?:荷官姓名|荷官|Dealer|dealerName|dealer_name)\s*[:：=]?\s*([A-Za-z\u4e00-\u9fff0-9_-]{1,30})",
            blob,
            flags=re.IGNORECASE,
        )
        if dealer_match:
            dealer = dealer_match.group(1)

        online_match = re.search(
            r"(?:在線人數|在线人数|Online|onlineCount|players|userCount)\s*[:：=]?\s*(\d+)",
            blob,
            flags=re.IGNORECASE,
        )
        if online_match:
            try:
                online_count = int(online_match.group(1))
            except Exception:
                pass

        countdown_match = re.search(
            r"(?:倒數計時|倒数计时|倒數|倒数|countdown|remain|timer)\s*[:：=]?\s*(\d+)",
            blob,
            flags=re.IGNORECASE,
        )
        if countdown_match:
            try:
                countdown = int(countdown_match.group(1))
            except Exception:
                pass

        # 從 network tables 補資料
        for t in network.get("tables", []) or []:
            if str(t.get("table_id", "")).upper() == table_id.upper() or not table_id:
                table_id = str(t.get("table_id") or table_id).upper()
                game_no = t.get("game_no") or game_no
                dealer = t.get("dealer") or dealer
                online_count = t.get("online_count") or online_count
                break

        status = "讀取中"
        low = blob.lower()
        if "可押注" in blob or "可下注" in blob or "betting" in low or "bet" in low:
            status = "可押注"
        elif "停止下注" in blob or "不可押注" in blob or "closed" in low:
            status = "停止下注"

        last_result = road[-1] if road else ""
        round_key = f"{table_id}:{game_no}:{len(road)}:{last_result}:{countdown}"
        return {
            "table_id": table_id,
            "game_no": game_no,
            "dealer": dealer,
            "online_count": online_count,
            "countdown": countdown,
            "status": status,
            "road": road,
            "round_key": round_key,
            "real_data": bool(game_no or dealer or online_count or road),
        }

    async def prepare_page(self, platform_key: str, hall_key: Optional[str] = None):
        playwright, browser, page, collector = await self._new_page(platform_key)
        if hall_key:
            platform = get_platform(platform_key)
            labels = platform.hall_labels.get(hall_key, [])
            if labels:
                await self._click_any_text(page, labels)
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                await page.wait_for_timeout(2500)
        return playwright, browser, page, collector

    async def list_tables(self, platform_key: str, hall_key: str) -> List[Dict[str, Any]]:
        playwright = None
        browser = None
        try:
            playwright, browser, page, collector = await self.prepare_page(platform_key, hall_key)
            dom = await self._extract_dom_dump(page)
            network = collector.extract()
            blob = "\n".join([dom.get("text", ""), "\n".join(dom.get("attrs", [])), _safe_json_text(network)])
            tables = self._merge_tables(
                network.get("tables", []),
                self._parse_tables_from_blob(dom.get("text", ""), "dom_text"),
                self._parse_tables_from_blob("\n".join(dom.get("attrs", [])), "dom_attrs"),
                self._parse_tables_from_blob(_safe_json_text(network), "network_blob"),
            )
            if not tables and ALLOW_DEFAULT_TABLE_IDS and DEFAULT_TABLE_IDS:
                return [{"table_id": x, "game_no": "", "dealer": "", "online_count": 0, "source": "fallback_default"} for x in DEFAULT_TABLE_IDS]
            return tables
        finally:
            await self._close(playwright, browser)

    async def read_table_data(self, platform_key: str, hall_key: str, table_id: str) -> Dict[str, Any]:
        playwright = None
        browser = None
        try:
            playwright, browser, page, collector = await self.prepare_page(platform_key, hall_key)
            await self._click_any_text(page, [table_id])
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            await page.wait_for_timeout(2500)

            dom = await self._extract_dom_dump(page)
            network = collector.extract()
            blob = "\n".join([dom.get("text", ""), "\n".join(dom.get("attrs", [])), _safe_json_text(network)])

            road: List[str] = []
            roads = network.get("roads", []) or []
            if roads:
                road = max(roads, key=len)
            if len(road) < 3:
                road = await self._extract_dom_road(page)
            if len(road) < 3:
                color_road = await self._extract_color_road(page, platform_key, table_id)
                if color_road:
                    road = color_road

            return self._parse_basic_data(blob=blob, fallback_table_id=table_id, road=road, network=network)
        finally:
            await self._close(playwright, browser)

    async def debug_page(self, platform_key: str, hall_key: str = "BACCARAT") -> Dict[str, Any]:
        playwright = None
        browser = None
        try:
            playwright, browser, page, collector = await self.prepare_page(platform_key, hall_key)
            dom = await self._extract_dom_dump(page)
            network = collector.extract()
            return {
                "ok": True,
                "platform": platform_key,
                "hall": hall_key,
                "title": await page.title(),
                "current_url": page.url,
                "frame_count": dom.get("frame_count", 0),
                "text_length": len(dom.get("text", "")),
                "text_preview": dom.get("text", "")[:1000],
                "attrs_preview": dom.get("attrs", [])[:80],
                "network_url_count": len(collector.urls),
                "network_urls_preview": collector.urls[-30:],
                "network_tables": network.get("tables", [])[:20],
                "network_roads_count": len(network.get("roads", [])),
                "message": "如果 text_length=0 且 network_tables 空，代表平台資料可能是 canvas/加密 ws/未登入或尚未進入大廳。",
            }
        finally:
            await self._close(playwright, browser)


class BaccaratLivePage:
    """常駐瀏覽器頁面，用於降低延遲。"""
    def __init__(self, platform_key: str, hall_key: str, table_id: str):
        self.platform_key = platform_key
        self.hall_key = hall_key
        self.table_id = table_id
        self.reader = BaccaratReader()
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.collector: Optional[NetworkCollector] = None
        self.ready = False

    async def start(self) -> None:
        self.playwright, self.browser, self.page, self.collector = await self.reader.prepare_page(self.platform_key, self.hall_key)
        if self.page:
            await self.reader._click_any_text(self.page, [self.table_id])
            await self.page.wait_for_timeout(2500)
        self.ready = True

    async def read(self) -> Dict[str, Any]:
        if not self.ready or not self.page or not self.collector:
            await self.start()
        assert self.page is not None
        assert self.collector is not None
        dom = await self.reader._extract_dom_dump(self.page)
        network = self.collector.extract()
        blob = "\n".join([dom.get("text", ""), "\n".join(dom.get("attrs", [])), _safe_json_text(network)])
        road: List[str] = []
        roads = network.get("roads", []) or []
        if roads:
            road = max(roads, key=len)
        if len(road) < 3:
            road = await self.reader._extract_dom_road(self.page)
        if len(road) < 3:
            color_road = await self.reader._extract_color_road(self.page, self.platform_key, self.table_id)
            if color_road:
                road = color_road
        return self.reader._parse_basic_data(blob, self.table_id, road, network)

    async def close(self) -> None:
        await self.reader._close(self.playwright, self.browser)
        self.ready = False
