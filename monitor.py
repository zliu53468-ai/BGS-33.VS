import asyncio
from typing import Any, Callable, Dict, Optional

from baccarat_reader import BaccaratReader
from config import AUTO_PUSH_NEW_ROUND, POLL_INTERVAL_SECONDS
from line_client import push_message
from line_messages import build_analysis_message, text_message
from predictor import predict
from session_store import get_session, update_session

PushCallback = Callable[[str, Dict[str, Any], Dict[str, Any]], None]


class MonitorManager:
    def __init__(self, reader: BaccaratReader) -> None:
        self.reader = reader
        self.tasks: Dict[str, asyncio.Task] = {}
        self.stop_flags: Dict[str, asyncio.Event] = {}

    def is_running(self, user_id: str) -> bool:
        task = self.tasks.get(user_id)
        return bool(task and not task.done())

    async def start(self, user_id: str) -> None:
        await self.stop(user_id)

        stop_event = asyncio.Event()
        self.stop_flags[user_id] = stop_event
        task = asyncio.create_task(self._run_loop(user_id, stop_event))
        self.tasks[user_id] = task

    async def stop(self, user_id: str) -> None:
        stop_event = self.stop_flags.get(user_id)
        if stop_event:
            stop_event.set()

        task = self.tasks.get(user_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except Exception:
                pass

        self.tasks.pop(user_id, None)
        self.stop_flags.pop(user_id, None)

    async def _run_loop(self, user_id: str, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                session = get_session(user_id)
                if not session.get("running"):
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                platform = session.get("platform")
                hall = session.get("hall")
                table_id = session.get("table_id")

                if not platform or not hall or not table_id:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                data = await self.reader.read_table_data(platform, hall, table_id)
                road = data.get("road", [])
                prediction = predict(road)
                new_key = data.get("round_key")
                old_key = session.get("last_round_key")

                update_session(
                    user_id,
                    game_no=data.get("game_no"),
                    dealer=data.get("dealer"),
                    online_count=data.get("online_count"),
                    road=road,
                    last_round_key=new_key,
                    running=True,
                    step="ANALYZING",
                )

                if AUTO_PUSH_NEW_ROUND and old_key and new_key and new_key != old_key:
                    push_message(user_id, [build_analysis_message(data, prediction)])

            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"monitor error user={user_id}: {exc}")
                if AUTO_PUSH_NEW_ROUND:
                    push_message(user_id, [text_message(f"自動讀取暫時失敗：{exc}")])

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
