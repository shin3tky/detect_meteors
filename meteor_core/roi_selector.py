#!/usr/bin/env python
#
# Detect Meteors CLI - ROI Selector
# © 2025 Shinichi Morita (shin3tky)
#

"""
ROI (Region of Interest) selection GUI for meteor detection.
"""

import logging
import math
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from .i18n import DEFAULT_LOCALE, get_message

logger = logging.getLogger(__name__)


def select_roi(
    image_data: np.ndarray, locale: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Polygon ROI selection with vertex editing.

    Args:
        image_data: Input image as numpy array (grayscale, uint16)

    Returns:
        Dictionary containing:
            - mask: Binary mask (numpy array, uint8)
            - polygon: List of vertices [[x1,y1], [x2,y2], ...]
            - bounding_rect: (x, y, w, h) tuple
        Or None if selection was cancelled or invalid.

    Controls:
        - Left click: add vertex
        - Esc: delete last vertex
        - Close by clicking the start circle (when >= 3 vertices)
        - 'q': cancel selection
    """
    if image_data is None:
        logger.error("No image data provided for ROI selection.")
        raise ValueError("image_data must not be None")

    if image_data.size == 0:
        logger.error("Empty image data provided for ROI selection.")
        raise ValueError("image_data must contain pixel data")

    disp_img = image_data.astype(np.float32)
    max_value = np.max(disp_img)
    if max_value == 0:
        logger.error("ROI selection image has zero maximum intensity.")
        raise ValueError("image_data must contain non-zero values")

    disp_img = disp_img / max_value
    # Brighten the image by 50% to improve visibility in dark shooting conditions
    disp_img = np.clip(disp_img * 1.5, 0, 1)
    disp_img = (disp_img * 255).astype(np.uint8)

    h, w = disp_img.shape
    scale_factor = 1.0

    if w > 1200:
        scale_factor = 1200 / w
        disp_w = int(w * scale_factor)
        disp_h = int(h * scale_factor)
        disp_img_resized = cv2.resize(disp_img, (disp_w, disp_h))
    else:
        disp_img_resized = disp_img

    display_img = cv2.cvtColor(disp_img_resized, cv2.COLOR_GRAY2BGR)
    resolved_locale = locale or DEFAULT_LOCALE
    window_name = get_message("ui.roi.window_title", locale=resolved_locale)

    logger.info("Starting ROI selection UI with image size (h=%d, w=%d)", h, w)
    print("\n" + get_message("ui.roi.mode.header", locale=resolved_locale))
    print(get_message("ui.roi.mode.instructions", locale=resolved_locale))

    points: List[Tuple[int, int]] = []
    mouse_pos: Optional[Tuple[int, int]] = None
    polygon_closed = False
    closable_threshold = 12
    closable_radius = 6
    cancelled = False

    def draw_canvas() -> np.ndarray:
        canvas = display_img.copy()

        if points:
            cv2.polylines(
                canvas, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2
            )
            for px, py in points:
                cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

        if mouse_pos and points:
            cv2.line(canvas, points[-1], mouse_pos, (255, 255, 0), 1)

        if len(points) >= 3:
            first = points[0]
            hover_distance = (
                math.hypot(mouse_pos[0] - first[0], mouse_pos[1] - first[1])
                if mouse_pos
                else None
            )
            if hover_distance is not None and hover_distance <= closable_threshold:
                cv2.circle(canvas, first, closable_radius, (0, 255, 255), -1)

        return canvas

    def on_mouse(event: int, x: int, y: int, *_) -> None:
        nonlocal mouse_pos, polygon_closed
        mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if (
                len(points) >= 3
                and math.hypot(x - points[0][0], y - points[0][1]) <= closable_threshold
            ):
                polygon_closed = True
            else:
                points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            canvas = draw_canvas()
            cv2.imshow(window_name, canvas)

            if polygon_closed:
                cv2.polylines(
                    canvas, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2
                )
                cv2.imshow(window_name, canvas)
                cv2.waitKey(300)
                break

            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC: delete last vertex
                if points:
                    points.pop()
            elif key == ord("q"):
                cancelled = True
                break
    except Exception as exc:  # pragma: no cover - UI safety
        logger.exception("Unexpected error in ROI selection loop: %s", exc)
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
        raise

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    if cancelled or len(points) < 3:
        logger.warning("ROI selection cancelled or insufficient points selected.")
        return None

    points_scaled = [
        (int(px / scale_factor), int(py / scale_factor)) for px, py in points
    ]
    polygon = np.array(points_scaled, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    bounding_rect = cv2.boundingRect(polygon)

    logger.info(
        "ROI selection completed with %d vertices and bounding box %s",
        len(points_scaled),
        bounding_rect,
    )
    return {"mask": mask, "polygon": polygon.tolist(), "bounding_rect": bounding_rect}


def create_roi_mask_from_polygon(
    polygon: List[List[int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create a binary ROI mask from a polygon.

    Args:
        polygon: List of vertices [[x1,y1], [x2,y2], ...]
        image_shape: (height, width) of the target image

    Returns:
        Binary mask as numpy array (uint8)
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask


def create_full_roi_mask(image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a full-image ROI mask (no region excluded).

    Args:
        image_shape: (height, width) of the target image

    Returns:
        Binary mask with all pixels set to 255
    """
    height, width = image_shape
    return np.full((height, width), 255, dtype=np.uint8)
