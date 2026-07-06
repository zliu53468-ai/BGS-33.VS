# 富百家 AI Pro 自動分析版

## 這版已修正

1. 關閉假桌號 fallback：沒有真實抓到桌號時，不會再顯示假 RB01。
2. `baccarat_reader.py` 加強 DOM / iframe / attribute / Network JSON 抓取。
3. `monitor.py` 使用常駐瀏覽器頁面，降低每次分析都重新開 Chromium 的延遲。
4. `predictor.py` 加入本地牌路判斷：長龍、斷龍風險、單跳、雙跳、成對、散盤觀望、歷史重複規律。
5. `deepseek_client.py` 可啟用 DeepSeek API 做獨立 AI 校準。
6. 前端改為：開始預測 → 選平台 → 選遊戲廳 → 選真實桌號 → 分析數據。

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
READER_WAIT_MS=8000
```

## DeepSeek 環境變數

```env
DEEPSEEK_ENABLED=true
DEEPSEEK_API_KEY=你的DeepSeek API Key
DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SECONDS=8
DEEPSEEK_WEIGHT=0.30
LOCAL_MODEL_WEIGHT=0.70
```

## Debug API

```text
/health
/api/debug/playwright
/api/debug/baccarat?platform=DG&hall=BACCARAT
/api/debug/tables?platform=DG&hall=BACCARAT
/api/debug/table?platform=DG&hall=BACCARAT&table_id=RB05
```

判斷是否真的爬到資料：

- `table_count > 0` 且 `source != fallback_default`：有抓到真實桌號。
- `road_length > 0`：有抓到牌路。
- `prediction.ai_used=true`：DeepSeek 有成功參與校準。

## LINE Webhook

```text
https://你的-render網址.onrender.com/webhook
```
