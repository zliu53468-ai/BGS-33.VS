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
    LOGIN_CHECK_ENABLED,
    LOGIN_EXPIRED_KEYWORDS,
    LOGIN_MIN_TEXT_LENGTH,
    READER_WAIT_MS,
    STRICT_TABLE_METADATA,
    TARGET_TABLE_ONLY,
    TABLE_CLICK_TIMEOUT_MS,
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


class LoginExpiredError(RuntimeError):
    def __init__(self, platform_key: str, message: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        self.platform_key = platform_key
        self.diagnostics = diagnostics or {}
        super().__init__(message)


def is_url_like_login_page(url: str) -> bool:
    u = (url or "").lower()
    return any(x in u for x in ["login", "signin", "sign-in", "auth", "passport"]) and not any(x in u for x in ["opengame", "game", "lobby"])


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

    def record_ws_payload(self, payload: Any) -> None:
        if not USE_NETWORK_READER:
            return
        try:
            if isinstance(payload, bytes):
                text = payload.decode("utf-8", errors="ignore")
            else:
                text = str(payload or "")
            if not text:
                return
            text = text[:200000]
            self.text_chunks.append(text)
            if len(self.text_chunks) > self.max_items:
                self.text_chunks = self.text_chunks[-self.max_items:]
            stripped = text.strip()
            if stripped.startswith(("{", "[")):
                try:
                    self.json_chunks.append(json.loads(stripped))
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

        def _attach_ws(ws):
            try:
                ws.on("framereceived", lambda payload: collector.record_ws_payload(payload))
                ws.on("framesent", lambda payload: collector.record_ws_payload(payload))
            except Exception:
                pass

        page.on("websocket", _attach_ws)
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
            safe_label = str(label).strip()
            # 先 exact，避免輸入 E5 時點到 E9 / E50 / 其他局號。
            for exact in (True, False):
                for frame in page.frames:
                    try:
                        loc = frame.get_by_text(safe_label, exact=exact)
                        if await loc.count() > 0:
                            await loc.first.click(timeout=TABLE_CLICK_TIMEOUT_MS)
                            await page.wait_for_timeout(max(800, min(READER_WAIT_MS, 3500)))
                            return True
                    except Exception:
                        continue
            for frame in page.frames:
                try:
                    loc = frame.locator(f"[aria-label*='{safe_label}'], [title*='{safe_label}'], [alt*='{safe_label}'], [data-name*='{safe_label}'], [data-title*='{safe_label}'], [data-table*='{safe_label}'], [data-table-id*='{safe_label}']")
                    if await loc.count() > 0:
                        await loc.first.click(timeout=TABLE_CLICK_TIMEOUT_MS)
                        await page.wait_for_timeout(max(800, min(READER_WAIT_MS, 3500)))
                        return True
                except Exception:
                    continue
        return False

    async def prepare_page(self, platform_key: str, hall_key: Optional[str] = None) -> Tuple[Any, Any, Page, NetworkCollector]:
        playwright, browser, page, collector = await self.new_browser_page(platform_key)
        await self.ensure_login_valid(page, collector, platform_key, stage="platform_entry")
        if hall_key:
            platform = get_platform(platform_key)
            labels = platform.hall_labels.get(hall_key, [])
            await self.click_any_text(page, labels)
            await self.ensure_login_valid(page, collector, platform_key, stage="hall_entry")
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

    def detect_login_expired(self, text: str = "", url: str = "", title: str = "", collector: Optional[NetworkCollector] = None) -> Dict[str, Any]:
        """
        判斷遊戲登入 token / session 是否失效。
        重點：不要讓後續邏輯把登入頁、錯誤頁、試玩結束頁誤判成桌廳頁。
        """
        if not LOGIN_CHECK_ENABLED:
            return {"login_expired": False, "login_ok": True, "reason": "login_check_disabled", "matched_keywords": []}

        combined = "\n".join([
            str(title or ""),
            str(url or ""),
            str(text or ""),
            collector.all_text() if collector else "",
        ])
        lowered = combined.lower()
        matched = []

        # 強特徵：網址已經跳回 login/auth，通常就是 token/session 無效。
        if is_url_like_login_page(url):
            matched.append("url_login_page")

        # 強特徵：錯誤頁/逾時/重新登入相關字眼。
        for kw in LOGIN_EXPIRED_KEYWORDS:
            if kw and kw in lowered:
                matched.append(kw)

        # 如果頁面文字極少，同時沒有 network/json，常見於 token 已失效後的空白頁。
        text_length = len(str(text or "").strip())
        network_count = len(collector.text_chunks) if collector else 0
        json_count = len(collector.json_chunks) if collector else 0
        maybe_blank = text_length < LOGIN_MIN_TEXT_LENGTH and network_count == 0 and json_count == 0

        expired = bool(matched) or maybe_blank
        reason = ""
        if matched:
            reason = "登入憑證/token 可能已失效，偵測到：" + ", ".join(matched[:6])
        elif maybe_blank:
            reason = "頁面文字與 Network 資料過少，可能是空白頁、token失效或平台阻擋。"
        else:
            reason = "login_ok"

        return {
            "login_expired": expired,
            "login_ok": not expired,
            "reason": reason,
            "matched_keywords": matched[:20],
            "text_length": text_length,
            "network_chunks": network_count,
            "json_chunks": json_count,
            "current_url": url,
            "title": title,
        }

    async def get_login_state(self, page: Page, collector: Optional[NetworkCollector] = None, text: str = "") -> Dict[str, Any]:
        try:
            title = await page.title()
        except Exception:
            title = ""
        if not text:
            try:
                text = await self.extract_visible_text(page)
            except Exception:
                text = ""
        return self.detect_login_expired(text=text, url=page.url, title=title, collector=collector)

    async def ensure_login_valid(self, page: Page, collector: Optional[NetworkCollector], platform_key: str, stage: str = "") -> None:
        state = await self.get_login_state(page, collector)
        if state.get("login_expired"):
            message = (
                f"{platform_key} 平台登入已失效或頁面未成功進入遊戲大廳。"
                f"請到 Render 更新 BACCARAT_URL_{platform_key} 的完整登入網址/token 後重新部署。"
            )
            if stage:
                message += f"（階段：{stage}）"
            message += f"\n原因：{state.get('reason')}"
            raise LoginExpiredError(platform_key, message, state)

    def login_expired_data(self, platform_key: str, table_id: str = "", diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        diagnostics = diagnostics or {}
        return {
            "table_id": table_id or "",
            "game_no": "",
            "dealer": "",
            "online_count": 0,
            "countdown": 0,
            "status": "登入失效",
            "road": [],
            "round_key": f"login_expired:{platform_key}:{table_id}",
            "source": "login_expired",
            "error": (
                f"{platform_key} 平台登入已失效或 token 過期，請更新 Render 環境變數 "
                f"BACCARAT_URL_{platform_key} 的完整登入網址。"
            ),
            "login_expired": True,
            "diagnostics": diagnostics,
        }

    def _json_item_matches_table(self, item: Dict[str, Any], table_id: str) -> bool:
        if not table_id:
            return True
        target = str(table_id or "").strip().upper()
        fields = [
            item.get("tableId"), item.get("table_id"), item.get("table"), item.get("tableNo"),
            item.get("tableName"), item.get("tableCode"), item.get("id"), item.get("gameNo"),
            item.get("game_no"), item.get("gameId"), item.get("roundId"), item.get("shoeId"),
        ]
        joined = " ".join(str(x or "").upper() for x in fields)
        return bool(target and target in joined)

    def extract_road_from_json(self, collector: Optional[NetworkCollector], table_id: str = "") -> List[str]:
        if not collector:
            return []

        def collect_from_value(v: Any, out: List[str]) -> None:
            if v is None:
                return
            if isinstance(v, str):
                s = v.strip().upper()
                if len(s) > 1 and all(ch in "BPT" for ch in s):
                    out.extend([normalize_side(ch) for ch in s if normalize_side(ch)])
                else:
                    side = normalize_side(s)
                    if side:
                        out.append(side)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        for kk in ["result", "winner", "side", "value", "type", "gameResult"]:
                            if kk in it:
                                collect_from_value(it.get(kk), out)
                    else:
                        collect_from_value(it, out)
            elif isinstance(v, dict):
                for kk in ["result", "winner", "side", "value", "type", "gameResult"]:
                    if kk in v:
                        collect_from_value(v.get(kk), out)

        road: List[str] = []
        target_found = False
        road_keys = {"result", "winner", "win", "side", "gameresult", "road", "roads", "bead", "history", "results"}

        # 優先找包含指定桌號/房號的 JSON object，避免把其他桌的紅藍綠混進來。
        for obj in collector.json_chunks[-160:]:
            for item in flatten_json_values(obj):
                if not isinstance(item, dict):
                    continue
                if table_id and not self._json_item_matches_table(item, table_id):
                    continue
                local: List[str] = []
                for k, v in item.items():
                    if str(k).lower() in road_keys:
                        collect_from_value(v, local)
                if local:
                    target_found = True
                    road.extend(local)

        if target_found:
            return road[-120:]

        # 沒有指定桌的 JSON 時才使用全域結果，避免太早放棄。
        if table_id and TARGET_TABLE_ONLY:
            return []

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if str(k).lower() in road_keys:
                        collect_from_value(v, road)
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        for item in collector.json_chunks[-80:]:
            walk(item)
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

    def _is_reasonable_table_id(self, table_id: str) -> bool:
        """
        v6：桌廳格式每個系統都不同，不能只接受 RB/R 開頭。
        允許 E5、E9、DG66、A12、R5037、RB05 等房號/桌號。
        只排除明顯不是桌號的系統文字、年份、純中文標籤與過長 token。
        """
        tid = str(table_id or "").strip().upper().replace(" ", "")
        if not tid:
            return False
        bad = {
            "TRUE", "FALSE", "NULL", "NONE", "READ", "TEXT", "HTML", "BODY",
            "DG", "MT", "SA", "T9", "GSA", "OB", "AB", "AI", "PRO",
            "SELECT", "BUTTON", "IMG", "LOGO", "BACCARAT", "BANKER", "PLAYER", "TIE",
            "ONLINE", "DEALER", "TABLE", "GAME", "ROOM", "ROAD", "RESULT",
        }
        if tid in bad:
            return False
        if re.fullmatch(r"20\d{2}", tid):
            return False
        if len(tid) > 30:
            return False
        if not re.search(r"\d", tid):
            return False
        # 常見房號/桌號：E5、E9、DG66、RB05、R5037、202607060020R5037
        if re.fullmatch(r"[A-Z]{1,4}\d{1,6}", tid):
            return True
        if re.fullmatch(r"\d{1,3}", tid):
            return True
        if re.fullmatch(r"20\d{8,}[A-Z0-9]*", tid):
            return True
        return bool(re.fullmatch(r"[A-Z0-9_-]{2,30}", tid))

    def _has_real_table_metadata(self, row: Dict[str, Any]) -> bool:
        if not STRICT_TABLE_METADATA:
            return True
        source = str(row.get("source") or "")
        if source in {"network", "game_no"}:
            return True
        if row.get("game_no") or row.get("dealer"):
            return True
        try:
            if int(row.get("online_count") or 0) > 0:
                return True
        except Exception:
            pass
        tid = str(row.get("table_id") or "").upper()
        # 只有符合常見真桌號格式才允許純文字來源通過。
        return bool(re.fullmatch(r"(RB|SB|CB|TB|DT)\d{1,3}", tid) or re.fullmatch(r"R\d{3,6}", tid))

    def parse_tables_from_text(self, text: str, collector: Optional[NetworkCollector] = None) -> List[Dict[str, Any]]:
        combined = text or ""
        if collector:
            combined += "\n" + collector.all_text()

        tables: Dict[str, Dict[str, Any]] = {}

        # 1. 明確桌號語意 + 各系統常見短房號。
        # v6：每個系統房號不同，允許 E5 / E9 / DG66 / A12 / R5037 / RB05。
        semantic_patterns = [
            r"(?:桌號|桌号|桌台|房號|房号|房間|房间|Table|Room|tableId|table_id|tableNo|tableName|tableCode)\s*[:：= ]+\s*([A-Za-z0-9_-]{1,30})",
            r"\b(RB\d{1,3}|SB\d{1,3}|CB\d{1,3}|TB\d{1,3}|DT\d{1,3})\b",
            r"\b(R\d{3,6})\b",
            r"\b([A-Z]{1,4}\d{1,4})\b",
            r"(?:百家樂|百家乐)\s*(\d{1,3})\s*(?:號桌|号桌|桌|房)?",
        ]
        for pat in semantic_patterns:
            for m in re.finditer(pat, combined, flags=re.IGNORECASE):
                val = m.group(1) if m.lastindex else m.group(0)
                val = str(val).strip().upper()
                if val.isdigit():
                    val = f"TABLE{val}"
                if not self._is_reasonable_table_id(val):
                    continue
                # 若只是從普通文字抽到，標記為 text_card，代表已抓到候選桌但 metadata 可能需點入後才取得。
                tables.setdefault(val, {"table_id": val, "game_no": "", "dealer": "", "online_count": 0, "source": "text_card"})

        # 2. 遊戲編號，例如 202607060020R5037，抽 R5037 作桌碼。
        game_nos = re.findall(r"20\d{8,}[A-Z0-9]*", combined, flags=re.IGNORECASE)
        for game_no in game_nos:
            m = re.search(r"([A-Z]{1,3}\d{3,6})$", game_no, flags=re.IGNORECASE)
            table_id = m.group(1).upper() if m else ""
            if not table_id or not self._is_reasonable_table_id(table_id):
                continue
            item = tables.setdefault(table_id, {"table_id": table_id, "game_no": "", "dealer": "", "online_count": 0, "source": "game_no"})
            item["source"] = "game_no"
            if not item.get("game_no"):
                item["game_no"] = game_no

        # 3. 從 JSON / WebSocket 中找 table 類資料，這是最可信的來源。
        if collector:
            for obj in collector.json_chunks[-120:]:
                for item in flatten_json_values(obj):
                    if not isinstance(item, dict):
                        continue
                    raw_id = (
                        item.get("tableId") or item.get("table_id") or item.get("table") or
                        item.get("tableNo") or item.get("tableName") or item.get("tableCode") or
                        item.get("table_id_str") or item.get("id")
                    )
                    raw_game = item.get("gameNo") or item.get("game_no") or item.get("gameId") or item.get("roundId") or item.get("shoeId")
                    raw_dealer = item.get("dealer") or item.get("dealerName") or item.get("dealer_name") or item.get("荷官姓名") or item.get("dealerNm")
                    raw_online = item.get("online") or item.get("onlineCount") or item.get("online_count") or item.get("players") or item.get("userCount")

                    # 必須像桌資料：有桌號/局號，且至少有一個 metadata 或是明確桌號格式。
                    if not raw_id and not raw_game:
                        continue
                    table_id = str(raw_id or "").strip().upper()
                    if not table_id and raw_game:
                        m = re.search(r"([A-Z]{1,3}\d{2,6})$", str(raw_game), flags=re.IGNORECASE)
                        table_id = m.group(1).upper() if m else ""
                    if not table_id or not self._is_reasonable_table_id(table_id):
                        continue

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

        filtered: List[Dict[str, Any]] = []
        for row in tables.values():
            if not self._is_reasonable_table_id(str(row.get("table_id") or "")):
                continue
            if not self._has_real_table_metadata(row):
                continue
            filtered.append(row)

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
            "source": "road_detected" if road else "no_road",
        }

    async def read_from_page(self, page: Page, collector: Optional[NetworkCollector], platform_key: str, hall_key: str, table_id: str, target_locked: bool = False) -> Dict[str, Any]:
        text = await self.extract_visible_text(page)
        road_source = "no_road"
        road = self.extract_road_from_json(collector, table_id=table_id)
        if road:
            road_source = "network_target"

        # 選擇桌後才允許用畫面/DOM掃牌路，避免還在大廳時抓到其他桌。
        allow_page_scan = target_locked or not TARGET_TABLE_ONLY

        if len(road) < 3 and allow_page_scan:
            dom_road = await self.extract_dom_road(page)
            if dom_road:
                road = dom_road
                road_source = "dom_after_table_click"
        if len(road) < 3 and allow_page_scan:
            color_road = await self.extract_color_road(page, platform_key, table_id)
            if color_road:
                road = color_road
                road_source = "color_after_table_click"

        data = self.parse_basic_data(text, collector, table_id, road)
        data["targeted"] = bool(target_locked)
        data["target_locked"] = bool(target_locked)
        data["source"] = road_source
        if TARGET_TABLE_ONLY and not target_locked and not road:
            data["status"] = "指定桌尚未定位"
            data["source"] = "target_not_clicked"
        return data

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
            target_locked = await self.click_any_text(page, [table_id])
            await page.wait_for_timeout(max(800, min(READER_WAIT_MS, 3500)))
            return await self.read_from_page(page, collector, platform_key, hall_key, table_id, target_locked=target_locked)
        finally:
            await self.close_browser(playwright, browser)
