# monitor.py
# -*- coding: utf-8 -*-

import asyncio
from typing import Any, Callable, Dict, Optional

from config import POLL_INTERVAL_SECONDS, AUTO_PUSH_NEW_ROUND
from baccarat_reader import BaccaratLivePage
from predictor import predict
from session_store import update_session, get_session


class MonitorManager:
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.live_pages: Dict[str, BaccaratLivePage] = {}
        self.latest: Dict[str, Dict[str, Any]] = {}
        self.on_push: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], Any]] = None

    def set_push_callback(self, cb: Callable[[str, Dict[str, Any], Dict[str, Any]], Any]) -> None:
        self.on_push = cb

    def is_running(self, user_id: str) -> bool:
        task = self.tasks.get(user_id)
        return bool(task and not task.done())

    async def start(self, user_id: str, platform: str, hall: str, table_id: str) -> Dict[str, Any]:
        await self.stop(user_id)
        live = BaccaratLivePage(platform, hall, table_id)
        self.live_pages[user_id] = live
        await live.start()
        data = await live.read()
        prediction = predict(data.get("road", []))
        self.latest[user_id] = {"data": data, "prediction": prediction}
        update_session(
            user_id,
            step="ANALYZING",
            running=True,
            platform=platform,
            hall=hall,
            table_id=table_id,
            game_no=data.get("game_no"),
            dealer=data.get("dealer"),
            online_count=data.get("online_count", 0),
            countdown=data.get("countdown", 0),
            status=data.get("status", "讀取中"),
            road=data.get("road", []),
            last_round_key=data.get("round_key"),
            last_prediction=prediction,
            real_data=data.get("real_data", False),
        )
        self.tasks[user_id] = asyncio.create_task(self._loop(user_id))
        return self.latest[user_id]

    async def _loop(self, user_id: str) -> None:
        while True:
            try:
                session = get_session(user_id)
                if not session.get("running"):
                    break
                live = self.live_pages.get(user_id)
                if not live:
                    break
                data = await live.read()
                prediction = predict(data.get("road", []))
                previous_key = session.get("last_round_key")
                current_key = data.get("round_key")
                changed = bool(current_key and current_key != previous_key)
                self.latest[user_id] = {"data": data, "prediction": prediction}
                update_session(
                    user_id,
                    game_no=data.get("game_no"),
                    dealer=data.get("dealer"),
                    online_count=data.get("online_count", 0),
                    countdown=data.get("countdown", 0),
                    status=data.get("status", "讀取中"),
                    road=data.get("road", []),
                    last_round_key=current_key,
                    last_prediction=prediction,
                    real_data=data.get("real_data", False),
                )
                if AUTO_PUSH_NEW_ROUND and changed and self.on_push:
                    maybe_coro = self.on_push(user_id, data, prediction)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                update_session(user_id, status=f"監控錯誤：{e}")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def read_now(self, user_id: str) -> Dict[str, Any]:
        live = self.live_pages.get(user_id)
        session = get_session(user_id)
        if not live:
            platform = session.get("platform")
            hall = session.get("hall")
            table_id = session.get("table_id")
            if not platform or not hall or not table_id:
                return {"error": "請先選擇平台、遊戲廳與桌號。"}
            return await self.start(user_id, platform, hall, table_id)
        data = await live.read()
        prediction = predict(data.get("road", []))
        self.latest[user_id] = {"data": data, "prediction": prediction}
        update_session(
            user_id,
            game_no=data.get("game_no"),
            dealer=data.get("dealer"),
            online_count=data.get("online_count", 0),
            countdown=data.get("countdown", 0),
            status=data.get("status", "讀取中"),
            road=data.get("road", []),
            last_round_key=data.get("round_key"),
            last_prediction=prediction,
            real_data=data.get("real_data", False),
        )
        return self.latest[user_id]

    async def stop(self, user_id: str) -> None:
        task = self.tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except Exception:
                pass
        live = self.live_pages.pop(user_id, None)
        if live:
            await live.close()
        update_session(user_id, running=False, step="STOPPED", status="已停止")
