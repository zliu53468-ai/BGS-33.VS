# baccarat_reader.py
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright, Page

from config import (
    ALLOW_DEFAULT_TABLE_IDS,
    DEFAULT_TABLE_IDS,
    HEADLESS,
    READER_WAIT_MS,
    USE_COLOR_READER,
    USE_DOM_READER,
    USE_NETWORK_READER,
    get_platform,
)
from color_reader import parse_road_from_screenshot

ROAD_SIDE_MAP = {
    "B": "B", "BANKER": "B", "莊": "B", "庄": "B",
    "P": "P", "PLAYER": "P", "閒": "P", "闲": "P",
    "T": "T", "TIE": "T", "和": "T",
}


class NetworkCollector:
    def __init__(self) -> None:
        self.text_chunks: List[str] = []
        self.json_chunks: List[Any] = []
        self.max_items = 200

    async def record_response(self, response) -> None:
        if not USE_NETWORK_READER:
            return
        try:
            req = response.request
            resource_type = req.resource_type
            url = response.url.lower()
            ctype = (response.headers.get("content-type") or "").lower()

            if resource_type not in {"xhr", "fetch", "websocket", "document"} and "json" not in ctype:
                return
            if any(skip in url for skip in [".png", ".jpg", ".jpeg", ".gif", ".css", ".woff", ".mp4"]):
                return

            body = await response.text()
            if not body:
                return
            body = body[:200000]
            self.text_chunks.append(body)
            if len(self.text_chunks) > self.max_items:
                self.text_chunks = self.text_chunks[-self.max_items:]

            if "json" in ctype or body.strip().startswith(("{", "[")):
                try:
                    self.json_chunks.append(json.loads(body))
                    if len(self.json_chunks) > self.max_items:
                        self.json_chunks = self.json_chunks[-self.max_items:]
                except Exception:
                    pass
        except Exception:
            return

    def all_text(self) -> str:
        return "\n".join(self.text_chunks[-60:])


def normalize_side(value: Any) -> str:
    v = str(value or "").strip().upper()
    if v in ROAD_SIDE_MAP:
        return ROAD_SIDE_MAP[v]
    if "BANKER" in v or "莊" in v or "庄" in v:
        return "B"
    if "PLAYER" in v or "閒" in v or "闲" in v:
        return "P"
    if "TIE" in v or v == "和":
        return "T"
    return ""


def flatten_json_values(obj: Any, depth: int = 0) -> List[Any]:
    if depth > 8:
        return []
    out: List[Any] = []
    if isinstance(obj, dict):
        out.append(obj)
        for v in obj.values():
            out.extend(flatten_json_values(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(flatten_json_values(item, depth + 1))
    else:
        out.append(obj)
    return out


class BaccaratReader:
    def __init__(self) -> None:
        self.viewport = {"width": 1365, "height": 900}
        self._playwright = None

    async def _start_playwright(self):
        return await async_playwright().start()

    async def new_browser_page(self, platform_key: str) -> Tuple[Any, Any, Page, NetworkCollector]:
        platform = get_platform(platform_key)
        playwright = await self._start_playwright()
        browser = await playwright.chromium.launch(
            headless=HEADLESS,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-setuid-sandbox",
            ],
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
        collector = NetworkCollector()
        page.on("response", lambda response: asyncio.create_task(collector.record_response(response)))
        await page.goto(platform.url, wait_until="domcontentloaded", timeout=60000)
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        await page.wait_for_timeout(READER_WAIT_MS)
        return playwright, browser, page, collector

    async def close_browser(self, playwright, browser) -> None:
        try:
            if browser:
                await browser.close()
        except Exception:
            pass
        try:
            if playwright:
                await playwright.stop()
        except Exception:
            pass

    async def click_any_text(self, page: Page, labels: List[str]) -> bool:
        for label in labels:
            if not label:
                continue
            for frame in page.frames:
                try:
                    loc = frame.get_by_text(label, exact=False)
                    if await loc.count() > 0:
                        await loc.first.click(timeout=2500)
                        await page.wait_for_timeout(READER_WAIT_MS)
                        return True
                except Exception:
                    continue
            for frame in page.frames:
                try:
                    loc = frame.locator(f"[aria-label*='{label}'], [title*='{label}'], [alt*='{label}'], [data-name*='{label}'], [data-title*='{label}']")
                    if await loc.count() > 0:
                        await loc.first.click(timeout=2500)
                        await page.wait_for_timeout(READER_WAIT_MS)
                        return True
                except Exception:
                    continue
        return False

    async def prepare_page(self, platform_key: str, hall_key: Optional[str] = None) -> Tuple[Any, Any, Page, NetworkCollector]:
        playwright, browser, page, collector = await self.new_browser_page(platform_key)
        if hall_key:
            platform = get_platform(platform_key)
            labels = platform.hall_labels.get(hall_key, [])
            await self.click_any_text(page, labels)
        return playwright, browser, page, collector

    async def extract_visible_text(self, page: Page) -> str:
        chunks: List[str] = []
        for frame in page.frames:
            try:
                body = await frame.locator("body").inner_text(timeout=3000)
                if body:
                    chunks.append(body)
            except Exception:
                pass

            script = """
            () => {
              const out = [];
              const nodes = Array.from(document.querySelectorAll('button,a,div,span,p,li,td,th,img,input,[aria-label],[title],[alt]'));
              for (const n of nodes) {
                const vals = [];
                vals.push(n.innerText || n.textContent || '');
                vals.push(n.getAttribute('aria-label') || '');
                vals.push(n.getAttribute('title') || '');
                vals.push(n.getAttribute('alt') || '');
                vals.push(n.getAttribute('value') || '');
                vals.push(n.getAttribute('data-table') || '');
                vals.push(n.getAttribute('data-table-id') || '');
                vals.push(n.getAttribute('data-id') || '');
                vals.push(n.getAttribute('data-name') || '');
                vals.push(n.getAttribute('data-game') || '');
                vals.push(n.getAttribute('data-result') || '');
                const cls = typeof n.className === 'string' ? n.className : '';
                vals.push(cls);
                const s = vals.map(x => String(x || '').trim()).filter(Boolean).join(' ');
                if (s) out.push(s);
              }
              return out.slice(0, 4000).join('\n');
            }
            """
            try:
                attrs = await frame.evaluate(script)
                if attrs:
                    chunks.append(attrs)
            except Exception:
                pass

        seen = set()
        unique: List[str] = []
        for c in chunks:
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return "\n".join(unique)

    def extract_road_from_json(self, collector: Optional[NetworkCollector]) -> List[str]:
        if not collector:
            return []
        candidates: List[str] = []
        keys = {"result", "winner", "win", "side", "banker", "player", "tie", "gameResult", "road", "roads", "bead", "history"}

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lk = str(k).lower()
                    if lk in {x.lower() for x in keys}:
                        if isinstance(v, str):
                            candidates.append(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, str):
                                    candidates.append(item)
                                elif isinstance(item, dict):
                                    for kk in ["result", "winner", "side", "value", "type"]:
                                        if kk in item:
                                            candidates.append(str(item.get(kk)))
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        for item in collector.json_chunks[-80:]:
            walk(item)

        road: List[str] = []
        for c in candidates:
            if not c:
                continue
            s = str(c).strip().upper()
            if len(s) > 1 and all(ch in "BPT" for ch in s):
                road.extend([normalize_side(ch) for ch in s if normalize_side(ch)])
            else:
                side = normalize_side(s)
                if side:
                    road.append(side)
        # 去除過度重複造成的雜訊，只取最後 120 手
        return road[-120:]

    async def extract_dom_road(self, page: Page) -> List[str]:
        if not USE_DOM_READER:
            return []
        script = """
        () => {
          const out = [];
          const nodes = Array.from(document.querySelectorAll('div,span,i,li,td,p,button,[data-result],[data-side],[data-value],[class]'));
          for (const n of nodes) {
            const rect = n.getBoundingClientRect();
            if (!rect || rect.width < 2 || rect.height < 2) continue;
            const text = (n.innerText || n.textContent || '').trim();
            const cls = typeof n.className === 'string' ? n.className : '';
            const style = n.getAttribute('style') || '';
            const data = [
              n.getAttribute('data-result') || '',
              n.getAttribute('data-type') || '',
              n.getAttribute('data-value') || '',
              n.getAttribute('data-side') || '',
              n.getAttribute('aria-label') || '',
              n.getAttribute('title') || '',
            ].join(' ');
            const combo = `${text} ${cls} ${style} ${data}`.toLowerCase();
            const roadHint = combo.includes('road') || combo.includes('bead') || combo.includes('result') || combo.includes('banker') || combo.includes('player') || combo.includes('tie') || combo.includes('莊') || combo.includes('庄') || combo.includes('閒') || combo.includes('闲') || combo.includes('和');
            if (!roadHint) continue;
            if (combo.includes('banker') || combo.includes('庄') || combo.includes('莊') || text === 'B') out.push('B');
            else if (combo.includes('player') || combo.includes('閒') || combo.includes('闲') || text === 'P') out.push('P');
            else if (combo.includes('tie') || combo.includes('和') || text === 'T') out.push('T');
          }
          return out.slice(-150);
        }
        """
        out: List[str] = []
        for frame in page.frames:
            try:
                vals = await frame.evaluate(script)
                if vals:
                    out.extend([v for v in vals if v in ("B", "P", "T")])
            except Exception:
                continue
        return out[-120:]

    async def extract_color_road(self, page: Page, platform_key: str, table_id: str) -> List[str]:
        if not USE_COLOR_READER:
            return []
        os.makedirs("tmp", exist_ok=True)
        path = f"tmp/{platform_key}_{table_id}.png"
        try:
            await page.screenshot(path=path, full_page=True)
            return parse_road_from_screenshot(path)
        except Exception:
            return []

    def parse_tables_from_text(self, text: str, collector: Optional[NetworkCollector] = None) -> List[Dict[str, Any]]:
        combined = text or ""
        if collector:
            combined += "\n" + collector.all_text()

        tables: Dict[str, Dict[str, Any]] = {}

        # 1. 常見桌號 / 局號中的桌號
        patterns = [
            r"\bRB\d{1,3}\b",
            r"\bSB\d{1,3}\b",
            r"\bCB\d{1,3}\b",
            r"\b[A-Z]{1,3}\d{1,4}\b",
            r"\bR\d{3,6}\b",
            r"(?:百家樂|百家乐)\s*(\d{1,3})\s*(?:號桌|号桌|桌)?",
            r"(?:桌號|桌号|Table|tableId|table_id)\s*[:：= ]+\s*([A-Za-z0-9_-]{2,30})",
        ]
        for pat in patterns:
            for m in re.finditer(pat, combined, flags=re.IGNORECASE):
                val = m.group(1) if m.lastindex else m.group(0)
                val = str(val).strip().upper()
                if val.isdigit():
                    val = f"TABLE{val}"
                if len(val) < 2:
                    continue
                tables.setdefault(val, {"table_id": val, "game_no": "", "dealer": "", "online_count": 0, "source": "text"})

        # 2. 遊戲編號，例如 202607060020R5037，抽 R5037 作桌碼備援
        game_nos = re.findall(r"20\d{8,}[A-Z0-9]*", combined, flags=re.IGNORECASE)
        for game_no in game_nos:
            m = re.search(r"([A-Z]{1,3}\d{3,6})$", game_no, flags=re.IGNORECASE)
            table_id = m.group(1).upper() if m else game_no[-6:].upper()
            item = tables.setdefault(table_id, {"table_id": table_id, "game_no": "", "dealer": "", "online_count": 0, "source": "game_no"})
            if not item.get("game_no"):
                item["game_no"] = game_no

        # 3. 從 JSON 中找 table 類資料
        if collector:
            for obj in collector.json_chunks[-80:]:
                for item in flatten_json_values(obj):
                    if not isinstance(item, dict):
                        continue
                    raw_id = item.get("tableId") or item.get("table_id") or item.get("table") or item.get("tableNo") or item.get("tableName") or item.get("id")
                    raw_game = item.get("gameNo") or item.get("game_no") or item.get("gameId") or item.get("roundId") or item.get("shoeId")
                    raw_dealer = item.get("dealer") or item.get("dealerName") or item.get("dealer_name") or item.get("荷官姓名")
                    raw_online = item.get("online") or item.get("onlineCount") or item.get("online_count") or item.get("players")
                    if raw_id or raw_game:
                        table_id = str(raw_id or "").strip().upper()
                        if not table_id and raw_game:
                            m = re.search(r"([A-Z]{1,3}\d{2,6})$", str(raw_game), flags=re.IGNORECASE)
                            table_id = m.group(1).upper() if m else str(raw_game)[-6:].upper()
                        if table_id:
                            row = tables.setdefault(table_id, {"table_id": table_id, "game_no": "", "dealer": "", "online_count": 0, "source": "network"})
                            row["source"] = "network"
                            if raw_game:
                                row["game_no"] = str(raw_game)
                            if raw_dealer:
                                row["dealer"] = str(raw_dealer)
                            try:
                                if raw_online is not None:
                                    row["online_count"] = int(raw_online)
                            except Exception:
                                pass

        out = list(tables.values())
        # 避免把太通用的年份、數字、空值當桌號
        filtered = []
        for t in out:
            tid = str(t.get("table_id", "")).upper()
            if not tid or tid in {"TRUE", "FALSE", "NULL", "NONE"}:
                continue
            if re.fullmatch(r"20\d{2}", tid):
                continue
            filtered.append(t)

        if not filtered and ALLOW_DEFAULT_TABLE_IDS:
            filtered = [{"table_id": tid, "game_no": "", "dealer": "", "online_count": 0, "source": "fallback_default"} for tid in DEFAULT_TABLE_IDS]
        return filtered[:30]

    def parse_basic_data(self, text: str, collector: Optional[NetworkCollector], fallback_table_id: str, road: List[str]) -> Dict[str, Any]:
        combined = (text or "") + "\n" + (collector.all_text() if collector else "")
        table_id = fallback_table_id
        game_no = ""
        dealer = ""
        online_count = 0
        countdown = 0

        m = re.search(r"(?:桌號|桌号|Table|tableId|table_id)\s*[:：= ]+\s*([A-Za-z0-9_-]{2,30})", combined, flags=re.IGNORECASE)
        if m:
            table_id = m.group(1).upper()

        gm = re.search(r"20\d{8,}[A-Z0-9]*", combined, flags=re.IGNORECASE)
        if gm:
            game_no = gm.group(0)

        dm = re.search(r"(?:荷官姓名|荷官|Dealer|dealerName)\s*[:：= ]+\s*([A-Za-z\u4e00-\u9fff0-9_-]{1,30})", combined, flags=re.IGNORECASE)
        if dm:
            dealer = dm.group(1)

        om = re.search(r"(?:在線人數|在线人数|Online|onlineCount)\s*[:：= ]+\s*(\d+)", combined, flags=re.IGNORECASE)
        if om:
            online_count = int(om.group(1))

        cm = re.search(r"(?:倒數計時|倒数计时|倒數|倒数|countdown)\s*[:：= ]+\s*(\d+)", combined, flags=re.IGNORECASE)
        if cm:
            countdown = int(cm.group(1))

        # JSON 補強
        if collector:
            for obj in collector.json_chunks[-80:]:
                for item in flatten_json_values(obj):
                    if not isinstance(item, dict):
                        continue
                    raw_id = item.get("tableId") or item.get("table_id") or item.get("table") or item.get("tableNo")
                    if raw_id and str(raw_id).upper() != str(fallback_table_id).upper():
                        continue
                    game_no = str(item.get("gameNo") or item.get("game_no") or item.get("gameId") or game_no or "")
                    dealer = str(item.get("dealer") or item.get("dealerName") or item.get("dealer_name") or dealer or "")
                    try:
                        online_count = int(item.get("online") or item.get("onlineCount") or item.get("online_count") or online_count or 0)
                    except Exception:
                        pass
                    try:
                        countdown = int(item.get("countdown") or item.get("countDown") or item.get("timer") or countdown or 0)
                    except Exception:
                        pass

        if "可押注" in combined or "可下注" in combined or "betting" in combined.lower():
            status = "可押注"
        elif "停止下注" in combined or "不可押注" in combined or "dealing" in combined.lower():
            status = "停止下注"
        else:
            status = "讀取中"

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
            "source": "network/dom/color" if road else "no_road",
        }

    async def read_from_page(self, page: Page, collector: Optional[NetworkCollector], platform_key: str, hall_key: str, table_id: str) -> Dict[str, Any]:
        text = await self.extract_visible_text(page)
        road = self.extract_road_from_json(collector)
        if len(road) < 3:
            road = await self.extract_dom_road(page)
        if len(road) < 3:
            color_road = await self.extract_color_road(page, platform_key, table_id)
            if color_road:
                road = color_road
        return self.parse_basic_data(text, collector, table_id, road)

    async def list_tables(self, platform_key: str, hall_key: str) -> List[Dict[str, Any]]:
        playwright = browser = None
        try:
            playwright, browser, page, collector = await self.prepare_page(platform_key, hall_key)
            text = await self.extract_visible_text(page)
            return self.parse_tables_from_text(text, collector)
        finally:
            await self.close_browser(playwright, browser)

    async def read_table_data(self, platform_key: str, hall_key: str, table_id: str) -> Dict[str, Any]:
        playwright = browser = None
        try:
            playwright, browser, page, collector = await self.prepare_page(platform_key, hall_key)
            await self.click_any_text(page, [table_id])
            await page.wait_for_timeout(READER_WAIT_MS)
            return await self.read_from_page(page, collector, platform_key, hall_key, table_id)
        finally:
            await self.close_browser(playwright, browser)
