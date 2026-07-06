# 富百家 AI Pro LINE Bot｜百家樂自動分析流程

這是一份可以直接放到 GitHub + Render 部署的完整專案。

## 功能流程

1. LINE 輸入「開始預測」
2. 顯示使用指南
3. 選擇平台：歐博真人 / DG真人 / Rebirth真人
4. 選擇遊戲廳：經典百家樂 / 龍虎門
5. 選擇桌號
6. 讀取桌台資料與牌路
7. predictor.py 回傳莊/閒/和機率與推薦
8. 按「繼續分析」可重新讀取
9. 按「結束分析」停止背景監控

## 重要安全提醒

平台網址若帶有 token，請放在 Render Environment Variables，不要提交到公開 GitHub。

## Render 設定

Build Command:

```bash
pip install -r requirements.txt && playwright install chromium
```

Start Command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## 必填環境變數

```env
LINE_CHANNEL_ACCESS_TOKEN=
LINE_CHANNEL_SECRET=
BACCARAT_URL_GSA=
BACCARAT_URL_DG=
BACCARAT_URL_REBIRTH=
```

## Webhook

Render 部署完成後，LINE Developers Webhook URL 設定：

```text
https://你的-render-url.onrender.com/webhook
```

## 測試 API

```bash
curl -X POST https://你的-render-url.onrender.com/api/test-analyze \
  -H "Content-Type: application/json" \
  -d '{"platform":"DG","hall":"BACCARAT","table_id":"RB05"}'
```

## 關於實際抓取準確度

不同百家樂系統可能使用 HTML、API、WebSocket、Canvas 或圖片牌路。
這份專案先提供通用讀取架構：

- 先嘗試 DOM 文字/class 讀取
- 若抓不到，再可開啟 OpenCV 顏色辨識
- 若平台有明確 API/WebSocket，後續可把 baccarat_reader.py 改成 API 直讀，會最穩
