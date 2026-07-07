# Baccarat AI LINE Bot v7

本版重點：

0. **登入/token失效偵測**：如果遊戲帳號被踢出、試玩連結過期、token失效或頁面跳回登入頁，系統會直接提示更新 Render 內的 BACCARAT_URL，不會再誤判成爬蟲或模型問題。

1. **手動房號優先**：選完平台與遊戲廳後，不再強制慢速掃描所有桌號，改成請使用者直接輸入房號/桌號，例如 `房號 E5`、`桌號 R5037`、`E5`。
2. **只針對指定桌監控**：程式會嘗試點入使用者輸入的房號/桌號，並且在 `TARGET_TABLE_ONLY=true` 時，未成功定位指定桌前不會用全頁 DOM/顏色牌路，避免抓到其他桌廳紅藍綠。
3. **自動掃描保留為輔助**：LINE 卡片與 LIFF 面板仍保留「自動掃描桌號」按鈕，但僅建議在平台 Network/DOM 有真實桌資料時使用。
4. **DeepSeek + 本地牌路模型**：`predictor.py` 保留本地長龍、單跳、雙跳、成對、散盤判斷，再由 DeepSeek 做校準。
5. **LINE loading animation**：按鈕後先顯示讀取動畫，降低使用者以為卡住的感覺。

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
PUBLIC_BASE_URL=https://your-render-service.onrender.com

BACCARAT_URL_GSA=
BACCARAT_URL_DG=
BACCARAT_URL_REBIRTH=

PLAYWRIGHT_BROWSERS_PATH=/opt/render/project/.playwright
HEADLESS=true
POLL_INTERVAL_SECONDS=5
FRONTEND_AUTO_POLL_MS=6000
AUTO_PUSH_NEW_ROUND=true

LINE_LOADING_ENABLED=true
LINE_LOADING_SECONDS=20

USE_DOM_READER=true
USE_NETWORK_READER=true
USE_COLOR_READER=true
ALLOW_DEFAULT_TABLE_IDS=false
STRICT_TABLE_METADATA=false
MANUAL_FIRST_TABLE_MODE=false
AUTO_SCAN_TABLES_ON_HALL=true
TABLE_CLICK_TIMEOUT_MS=2500
TARGET_TABLE_ONLY=true
READER_WAIT_MS=3500
```

## 顏色辨識設定

如果 DOM / Network 抓不到牌路，再開：

```env
USE_COLOR_READER=true
```

建議先設定指定桌畫面內的牌路 ROI：

```env
TARGET_ROAD_ROI_X=0
TARGET_ROAD_ROI_Y=0
TARGET_ROAD_ROI_W=0
TARGET_ROAD_ROI_H=0
```

v6 預設會在「成功點入你選擇的牌桌」後才進行紅藍綠掃描。若你發現牌路抓錯，請優先設定 `TARGET_ROAD_ROI_X/Y/W/H`，讓程式只掃描指定桌內的牌路區塊。

## DeepSeek

```env
DEEPSEEK_ENABLED=true
DEEPSEEK_API_KEY=
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
/api/debug/table?platform=DG&hall=BACCARAT&table_id=E5
```


## v6 變更重點

- 預設改為「選遊戲廳後自動掃描整個遊戲廳牌桌」。
- 桌號格式放寬，可抓 E5 / E9 / DG66 / A12 / R5037 / RB05 等不同系統格式。
- 選擇牌桌後才允許畫面紅藍綠辨識，避免在大廳抓到其他桌牌路。
- 仍保留手動房號備援，但不是主要流程。


## v7 登入 / token 失效偵測

請在 Render 保留：

```env
LOGIN_CHECK_ENABLED=true
LOGIN_MIN_TEXT_LENGTH=80
```

如果平台顯示特殊錯誤文字，可以把它加進：

```env
LOGIN_EXPIRED_KEYWORDS=token expired,請重新登入,試玩已結束,登入失效
```

測試時看：

```text
/api/debug/baccarat?platform=DG&hall=BACCARAT
```

如果回傳 `login_expired=true`，代表不是 predictor 或 LINE 錯，而是該平台登入網址 / token / 試玩 session 已失效，需要更新 Render 的 `BACCARAT_URL_DG`、`BACCARAT_URL_GSA` 或對應平台網址。
