from typing import List

import cv2
import numpy as np

from config import CELL_H, CELL_W, ROAD_COLS, ROAD_ROWS, ROAD_X, ROAD_Y


def detect_cell_color(cell_img) -> str:
    """
    紅色 = B 莊
    藍色 = P 閒
    綠色 = T 和
    空白 = ""
    """
    if cell_img is None or cell_img.size == 0:
        return ""

    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    red_mask += cv2.inRange(hsv, np.array([168, 70, 70]), np.array([180, 255, 255]))

    blue_mask = cv2.inRange(hsv, np.array([88, 60, 60]), np.array([135, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([88, 255, 255]))

    scores = {
        "B": cv2.countNonZero(red_mask),
        "P": cv2.countNonZero(blue_mask),
        "T": cv2.countNonZero(green_mask),
    }

    result = max(scores, key=scores.get)

    if scores[result] < 25:
        return ""

    return result


def parse_road_from_screenshot(image_path: str) -> List[str]:
    img = cv2.imread(image_path)

    if img is None:
        return []

    h, w = img.shape[:2]
    road: List[str] = []

    for col in range(ROAD_COLS):
        for row in range(ROAD_ROWS):
            x1 = ROAD_X + col * CELL_W
            y1 = ROAD_Y + row * CELL_H
            x2 = x1 + CELL_W
            y2 = y1 + CELL_H

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            cell = img[y1:y2, x1:x2]
            value = detect_cell_color(cell)

            if value:
                road.append(value)

    return road
