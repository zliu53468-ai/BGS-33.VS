# color_reader.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import cv2
import numpy as np

from config import (
    AUTO_COLOR_FULL_SCAN,
    CELL_H,
    CELL_W,
    COLOR_MAX_AREA,
    COLOR_MIN_AREA,
    ROAD_AUTO_DETECT,
    ROAD_COLS,
    ROAD_ROWS,
    ROAD_ROI_H,
    ROAD_ROI_W,
    ROAD_ROI_X,
    ROAD_ROI_Y,
    ROAD_X,
    ROAD_Y,
    TARGET_ROAD_ROI_H,
    TARGET_ROAD_ROI_W,
    TARGET_ROAD_ROI_X,
    TARGET_ROAD_ROI_Y,
)


def _mask_scores(hsv) -> Dict[str, int]:
    red_mask = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255])) + \
        cv2.inRange(hsv, np.array([168, 60, 60]), np.array([180, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([88, 45, 45]), np.array([138, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([88, 255, 255]))
    return {
        "B": cv2.countNonZero(red_mask),
        "P": cv2.countNonZero(blue_mask),
        "T": cv2.countNonZero(green_mask),
    }


def detect_cell_color(cell_img) -> str:
    """紅色=莊B、藍色=閒P、綠色=和T。"""
    if cell_img is None or getattr(cell_img, "size", 0) == 0:
        return ""

    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    scores = _mask_scores(hsv)
    side = max(scores, key=scores.get)
    return side if scores[side] >= 20 else ""


def _combined_color_mask(hsv):
    red = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255])) + \
        cv2.inRange(hsv, np.array([168, 60, 60]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([88, 45, 45]), np.array([138, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([88, 255, 255]))
    return cv2.bitwise_or(cv2.bitwise_or(red, blue), green)


def _dedupe_points(points: List[Dict], min_dist: int = 8) -> List[Dict]:
    out: List[Dict] = []
    for p in sorted(points, key=lambda x: x.get("area", 0), reverse=True):
        cx, cy = p["cx"], p["cy"]
        if any(abs(cx - q["cx"]) <= min_dist and abs(cy - q["cy"]) <= min_dist for q in out):
            continue
        out.append(p)
    return out


def _sort_bead_road(points: List[Dict]) -> List[Dict]:
    """
    多數珠盤路是直欄由上到下，接著往右一欄。
    用 x 分欄，再依 y 排序。
    """
    if not points:
        return []

    # 用中位數寬度估計欄距，避免同一欄被切太細
    widths = [max(8, int(p.get("w", 14))) for p in points]
    col_tol = max(10, int(np.median(widths) * 0.85)) if widths else 14

    cols: List[List[Dict]] = []
    for p in sorted(points, key=lambda z: z["cx"]):
        placed = False
        for col in cols:
            avg_x = sum(q["cx"] for q in col) / len(col)
            if abs(p["cx"] - avg_x) <= col_tol:
                col.append(p)
                placed = True
                break
        if not placed:
            cols.append([p])

    ordered: List[Dict] = []
    for col in cols:
        ordered.extend(sorted(col, key=lambda z: z["cy"]))
    return ordered


def auto_detect_road_from_screenshot(image_path: str) -> List[str]:
    """
    自動尋找截圖內紅/藍/綠圓點。
    建議設定 ROAD_ROI_X/Y/W/H，只掃描牌路區塊，避免抓到背景圖示或按鈕顏色。
    若沒設定 ROI，預設不做全圖掃描，避免誤判；要全圖掃描請設 AUTO_COLOR_FULL_SCAN=true。
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    full_h, full_w = img.shape[:2]

    roi_x = TARGET_ROAD_ROI_X if TARGET_ROAD_ROI_W > 0 and TARGET_ROAD_ROI_H > 0 else ROAD_ROI_X
    roi_y = TARGET_ROAD_ROI_Y if TARGET_ROAD_ROI_W > 0 and TARGET_ROAD_ROI_H > 0 else ROAD_ROI_Y
    roi_w = TARGET_ROAD_ROI_W if TARGET_ROAD_ROI_W > 0 and TARGET_ROAD_ROI_H > 0 else ROAD_ROI_W
    roi_h = TARGET_ROAD_ROI_H if TARGET_ROAD_ROI_W > 0 and TARGET_ROAD_ROI_H > 0 else ROAD_ROI_H

    if roi_w > 0 and roi_h > 0:
        x1 = max(0, min(roi_x, full_w - 1))
        y1 = max(0, min(roi_y, full_h - 1))
        x2 = max(x1 + 1, min(x1 + roi_w, full_w))
        y2 = max(y1 + 1, min(y1 + roi_h, full_h))
        roi = img[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    elif AUTO_COLOR_FULL_SCAN:
        roi = img
        offset_x = offset_y = 0
    else:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _combined_color_mask(hsv)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: List[Dict] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < COLOR_MIN_AREA or area > COLOR_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5 or w > 80 or h > 80:
            continue
        ratio = w / max(1, h)
        if ratio < 0.45 or ratio > 2.2:
            continue
        peri = cv2.arcLength(cnt, True)
        roundness = (4 * math.pi * area / (peri * peri)) if peri > 0 else 0
        if roundness < 0.35:
            continue

        pad = 2
        cell = roi[max(0, y - pad):min(roi.shape[0], y + h + pad), max(0, x - pad):min(roi.shape[1], x + w + pad)]
        side = detect_cell_color(cell)
        if not side:
            continue
        points.append({
            "side": side,
            "cx": int(x + w / 2 + offset_x),
            "cy": int(y + h / 2 + offset_y),
            "w": int(w),
            "h": int(h),
            "area": float(area),
        })

    points = _dedupe_points(points)
    ordered = _sort_bead_road(points)
    return [p["side"] for p in ordered]


def parse_road_from_screenshot(
    image_path: str,
    road_x: Optional[int] = None,
    road_y: Optional[int] = None,
    cell_w: Optional[int] = None,
    cell_h: Optional[int] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
) -> List[str]:
    road_x = ROAD_X if road_x is None else road_x
    road_y = ROAD_Y if road_y is None else road_y
    cell_w = CELL_W if cell_w is None else cell_w
    cell_h = CELL_H if cell_h is None else cell_h
    cols = ROAD_COLS if cols is None else cols
    rows = ROAD_ROWS if rows is None else rows

    img = cv2.imread(image_path)
    if img is None:
        return []

    # 固定格子模式：必須有實際 ROAD_X / ROAD_Y，0 代表尚未校正，不自動從左上角亂切。
    if road_x > 0 and road_y > 0 and cols > 0 and rows > 0:
        h, w = img.shape[:2]
        out: List[str] = []
        for col in range(cols):
            for row in range(rows):
                x1 = road_x + col * cell_w
                y1 = road_y + row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue
                value = detect_cell_color(img[y1:y2, x1:x2])
                if value:
                    out.append(value)
        if out:
            return out

    if ROAD_AUTO_DETECT:
        return auto_detect_road_from_screenshot(image_path)

    return []
