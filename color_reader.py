# color_reader.py
# -*- coding: utf-8 -*-

from typing import List, Optional
import cv2
import numpy as np

from config import ROAD_X, ROAD_Y, CELL_W, CELL_H, ROAD_COLS, ROAD_ROWS


def detect_cell_color(cell_img) -> str:
    """
    紅色 = B / 莊
    藍色 = P / 閒
    綠色 = T / 和
    """
    if cell_img is None or cell_img.size == 0:
        return ""

    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 70, 70])
    red_upper1 = np.array([12, 255, 255])
    red_lower2 = np.array([168, 70, 70])
    red_upper2 = np.array([180, 255, 255])

    blue_lower = np.array([88, 60, 60])
    blue_upper = np.array([135, 255, 255])

    green_lower = np.array([35, 50, 50])
    green_upper = np.array([88, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    scores = {
        "B": cv2.countNonZero(red_mask),
        "P": cv2.countNonZero(blue_mask),
        "T": cv2.countNonZero(green_mask),
    }
    result = max(scores, key=scores.get)

    # 太低代表該格可能是空格或背景
    if scores[result] < 25:
        return ""

    return result


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

    height, width = img.shape[:2]
    results: List[str] = []

    for col in range(cols):
        for row in range(rows):
            x1 = road_x + col * cell_w
            y1 = road_y + row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue

            value = detect_cell_color(img[y1:y2, x1:x2])
            if value:
                results.append(value)

    return results
