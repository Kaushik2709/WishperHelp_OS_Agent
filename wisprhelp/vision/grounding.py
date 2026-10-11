"""Coordinate Grounding, Bounding Box Normalization & DPI Multi-Monitor Mapper.

Translates between normalized VLM model coordinates ([0..1000] or [0.0..1.0]),
downscaled frame coordinates, and actual OS pixel coordinates.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from wisprhelp.vision.capture import ScreenFrame

logger = logging.getLogger(__name__)


class CoordinateGrounding:
    """Production-grade visual coordinate grounding and mapping engine."""

    @staticmethod
    def normalized_1000_to_pixels(
        norm_x: int,
        norm_y: int,
        screen_width: int,
        screen_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Tuple[int, int]:
        """Converts Anthropic/OmniParser [0..1000] coordinates to screen pixels."""
        clamped_x = max(0, min(1000, norm_x))
        clamped_y = max(0, min(1000, norm_y))

        px = int(offset_x + (clamped_x / 1000.0) * screen_width)
        py = int(offset_y + (clamped_y / 1000.0) * screen_height)

        return px, py

    @staticmethod
    def normalized_float_to_pixels(
        rel_x: float,
        rel_y: float,
        screen_width: int,
        screen_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Tuple[int, int]:
        """Converts relative float [0.0..1.0] coordinates to screen pixels."""
        clamped_x = max(0.0, min(1.0, rel_x))
        clamped_y = max(0.0, min(1.0, rel_y))

        px = int(offset_x + clamped_x * screen_width)
        py = int(offset_y + clamped_y * screen_height)

        return px, py

    @staticmethod
    def frame_coords_to_screen_pixels(
        frame_x: int,
        frame_y: int,
        frame: ScreenFrame,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Tuple[int, int]:
        """Maps coordinates from a downscaled ScreenFrame back to true monitor pixels."""
        orig_x = int(frame_x * frame.scale_x)
        orig_y = int(frame_y * frame.scale_y)

        # Clamp within original bounds
        clamped_x = max(0, min(frame.original_width - 1, orig_x))
        clamped_y = max(0, min(frame.original_height - 1, orig_y))

        return offset_x + frame.offset_x + clamped_x, offset_y + frame.offset_y + clamped_y

    @staticmethod
    def bbox_to_center(
        box: Tuple[int, int, int, int],  # (x, y, w, h) or (ymin, xmin, ymax, xmax)
        format: str = "xywh",
    ) -> Tuple[int, int]:
        """Computes center point (cx, cy) from bounding box."""
        if format == "xywh":
            x, y, w, h = box
            return x + w // 2, y + h // 2
        elif format == "ymin_xmin_ymax_xmax":
            ymin, xmin, ymax, xmax = box
            return xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2
        else:
            raise ValueError(f"Unknown bbox format: {format}")
