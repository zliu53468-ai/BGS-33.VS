# color_reader.py
# -*- coding: utf-8 -*-

from typing import List, Optional
import cv2
import numpy as np

from config import ROAD_X, ROAD_Y, CELL_W, CELL_H, ROAD_COLS, ROAD_ROWS


def detect_cell_color(cell_img) -> str:
    """紅色=莊B、藍色=閒P、綠色=和T。"""
    if cell_img is None or getattr(cell_img, "size", 0) == 0:
        return ""

    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255])) + \
        cv2.inRange(hsv, np.array([168, 60, 60]), np.array([180, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([88, 45, 45]), np.array([138, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([88, 255, 255]))

    scores = {
        "B": cv2.countNonZero(red_mask),
        "P": cv2.countNonZero(blue_mask),
        "T": cv2.countNonZero(green_mask),
    }
    side = max(scores, key=scores.get)
    return side if scores[side] >= 20 else ""


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

    if road_x < 0 or road_y < 0:
        return []

    img = cv2.imread(image_path)
    if img is None:
        return []

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
    return out
