# monitor.py
# -*- coding: utf-8 -*-

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from baccarat_reader import BaccaratReader, LoginExpiredError
from config import AUTO_PUSH_NEW_ROUND, POLL_INTERVAL_SECONDS
from predictor import predict
from session_store import update_session

OnUpdate = Callable[[str, Dict[str, Any], Dict[str, Any]], Awaitable[None]]


class MonitorSession:
    def __init__(self, user_id: str, platform: str, hall: str, table_id: str, on_update: Optional[OnUpdate] = None) -> None:
        self.user_id = user_id
        self.platform = platform
        self.hall = hall
        self.table_id = table_id
        self.on_update = on_update
        self.reader = BaccaratReader()
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.latest_data: Optional[Dict[str, Any]] = None
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self.last_round_key: Optional[str] = None
        self.playwright = None
        self.browser = None
        self.page = None
        self.collector = None
        self.target_locked = False

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except Exception:
                pass
        await self._close()

    async def _close(self) -> None:
        try:
            await self.reader.close_browser(self.playwright, self.browser)
        except Exception:
            pass
        self.playwright = self.browser = self.page = self.collector = None

    async def _open(self) -> None:
        self.playwright, self.browser, self.page, self.collector = await self.reader.prepare_page(self.platform, self.hall)
        self.target_locked = await self.reader.click_any_text(self.page, [self.table_id])

    async def refresh_once(self) -> Dict[str, Any]:
        if not self.page:
            await self._open()
        if not self.target_locked:
            self.target_locked = await self.reader.click_any_text(self.page, [self.table_id])
        data = await self.reader.read_from_page(self.page, self.collector, self.platform, self.hall, self.table_id, target_locked=self.target_locked)
        road = data.get("road", [])
        prediction = predict(road)
        self.latest_data = data
        self.latest_prediction = prediction
        self.last_round_key = data.get("round_key")
        update_session(
            self.user_id,
            step="ANALYZING",
            platform=self.platform,
            hall=self.hall,
            table_id=self.table_id,
            game_no=data.get("game_no"),
            dealer=data.get("dealer"),
            online_count=data.get("online_count"),
            countdown=data.get("countdown"),
            status=data.get("status"),
            road=road,
            last_round_key=self.last_round_key,
            last_prediction=prediction,
            last_data=data,
            running=True,
        )
        return {"data": data, "prediction": prediction}

    async def _run(self) -> None:
        try:
            await self._open()
            while self.running:
                try:
                    if not self.target_locked:
                        self.target_locked = await self.reader.click_any_text(self.page, [self.table_id])
                    data = await self.reader.read_from_page(self.page, self.collector, self.platform, self.hall, self.table_id, target_locked=self.target_locked)
                    road = data.get("road", [])
                    prediction = predict(road)
                    round_key = data.get("round_key")

                    changed = round_key and round_key != self.last_round_key
                    first = self.last_round_key is None
                    self.latest_data = data
                    self.latest_prediction = prediction
                    self.last_round_key = round_key

                    update_session(
                        self.user_id,
                        step="ANALYZING",
                        platform=self.platform,
                        hall=self.hall,
                        table_id=self.table_id,
                        game_no=data.get("game_no"),
                        dealer=data.get("dealer"),
                        online_count=data.get("online_count"),
                        countdown=data.get("countdown"),
                        status=data.get("status"),
                        road=road,
                        last_round_key=round_key,
                        last_prediction=prediction,
                        last_data=data,
                        running=True,
                    )

                    if self.on_update and (first or (AUTO_PUSH_NEW_ROUND and changed)):
                        await self.on_update(self.user_id, data, prediction)

                except asyncio.CancelledError:
                    raise
                except LoginExpiredError as e:
                    data = self.reader.login_expired_data(self.platform, self.table_id, e.diagnostics)
                    update_session(self.user_id, status="登入失效", running=False, last_data=data)
                    self.running = False
                    if self.on_update:
                        await self.on_update(self.user_id, data, {"recommend": "觀望", "reason": str(e), "ai_used": False})
                    break
                except Exception as e:
                    update_session(self.user_id, status=f"監控錯誤：{e}", running=self.running)
                    if self.on_update:
                        await self.on_update(self.user_id, {"error": str(e), "table_id": self.table_id, "road": []}, {"recommend": "觀望", "reason": str(e)})
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        finally:
            await self._close()


class MonitorManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, MonitorSession] = {}

    async def start(self, user_id: str, platform: str, hall: str, table_id: str, on_update: Optional[OnUpdate] = None) -> MonitorSession:
        await self.stop(user_id)
        session = MonitorSession(user_id, platform, hall, table_id, on_update=on_update)
        self.sessions[user_id] = session
        await session.start()
        return session

    async def stop(self, user_id: str) -> None:
        session = self.sessions.pop(user_id, None)
        if session:
            await session.stop()

    def get(self, user_id: str) -> Optional[MonitorSession]:
        return self.sessions.get(user_id)

    async def refresh_once(self, user_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(user_id)
        if not session:
            return None
        return await session.refresh_once()
