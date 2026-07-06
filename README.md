# 富百家 AI Pro - 上線覆蓋版 v3

## 已更新重點

1. `app.py`：結束分析會回到主選單，選廳/選桌改成先回覆讀取中，避免 LINE 按鈕長時間卡住。
2. `monitor.py`：選桌後建立常駐監控，降低每次都重新開瀏覽器的延遲。
3. `baccarat_reader.py`：關閉假桌號 fallback，抓不到真實桌號時回傳空陣列；加強 DOM / iframe / 屬性 / Network JSON 解析。
4. `predictor.py`：本地牌路模型 + DeepSeek AI 校準，支援長龍、斷龍、單跳、雙跳、成對、散盤觀望。
5. `deepseek_client.py`：提供 `DeepSeekClient.analyze_road()` 給 predictor.py 呼叫。
6. `line_messages.py`：分析卡顯示牌路型態、AI狀態、推薦與原因。
7. `static/index.html`：LIFF 改成自動流程版，不再是手動輸入 B/P/T 版。

## Render Build Command

```bash
pip install --upgrade pip && pip install -r requirements.txt && python -m playwright install --with-deps chromium
```

## Render Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## 必填環境變數

```env
LINE_CHANNEL_ACCESS_TOKEN=
LINE_CHANNEL_SECRET=
LIFF_ID=
PUBLIC_BASE_URL=https://你的-render網址.onrender.com

BACCARAT_URL_GSA=
BACCARAT_URL_DG=
BACCARAT_URL_REBIRTH=

PLAYWRIGHT_BROWSERS_PATH=/opt/render/project/.playwright
HEADLESS=true
POLL_INTERVAL_SECONDS=5
FRONTEND_AUTO_POLL_MS=6000
AUTO_PUSH_NEW_ROUND=true

USE_DOM_READER=true
USE_NETWORK_READER=true
USE_COLOR_READER=false
ALLOW_DEFAULT_TABLE_IDS=false
READER_WAIT_MS=3500
```

## DeepSeek 可選環境變數

```env
DEEPSEEK_ENABLED=true
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SECONDS=8
DEEPSEEK_WEIGHT=0.30
LOCAL_MODEL_WEIGHT=0.70
```

## 測試網址

```text
/health
/api/debug/playwright
/api/debug/baccarat?platform=DG&hall=BACCARAT
/api/debug/tables?platform=DG&hall=BACCARAT
/api/debug/table?platform=DG&hall=BACCARAT&table_id=真實桌號
```

`ALLOW_DEFAULT_TABLE_IDS=false` 時，若 `/api/debug/tables` 的 `table_count=0`，代表尚未讀取到真實桌號，不會再用 RB01 假資料混淆判斷。
