"""High-Speed Screen Capture, Token-Efficient Downscaling & Perceptual Diffing.

Uses mss for sub-30ms screen acquisition, PIL for aspect-ratio-preserving
downscaling, and 64-bit dHash for perceptual UI difference detection.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ScreenFrame:
    """Holds captured frame metadata and downscaling calibration."""
    image: Image.Image
    width: int
    height: int
    original_width: int
    original_height: int
    scale_x: float  # original_width / width
    scale_y: float  # original_height / height
    dhash: int
    timestamp: float

    def to_base64_jpeg(self, quality: int = 80) -> str:
        """Converts frame to compact base64 JPEG string for VLM payload."""
        buffer = io.BytesIO()
        img_to_save = self.image
        if img_to_save.mode != "RGB":
            img_to_save = img_to_save.convert("RGB")
        img_to_save.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def to_bytes(self, format: str = "JPEG", quality: int = 80) -> bytes:
        """Converts frame to in-memory bytes."""
        buffer = io.BytesIO()
        img_to_save = self.image
        if format.upper() in ("JPEG", "JPG") and img_to_save.mode != "RGB":
            img_to_save = img_to_save.convert("RGB")
        img_to_save.save(buffer, format=format, quality=quality, optimize=True)
        return buffer.getvalue()


class ScreenCapture:
    """Production-grade screen capture and perceptual diffing engine."""

    def __init__(self, max_dimension: int = 1920) -> None:
        self.max_dimension = max_dimension
        self._sct = None

    def _get_sct(self):
        if self._sct is None:
            try:
                import mss

                self._sct = getattr(mss, "MSS", mss.mss)()
            except Exception as exc:
                logger.warning("Failed to initialize mss screen grabber: %s", exc)
                self._sct = False
        return self._sct

    @staticmethod
    def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
        """Computes a 64-bit difference hash (dHash) for fast perceptual diffing.

        Time complexity: O(hash_size^2) ~ O(64) operations (< 1ms).
        Space complexity: O(1) in-memory tiny buffer.
        """
        # Resize to (hash_size + 1, hash_size) grayscale
        resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        if hasattr(resized, "get_flattened_data"):
            pixels = list(resized.get_flattened_data())
        else:
            pixels = list(resized.getdata())

        diff_bits = 0
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                left_pixel = pixels[row_start + col]
                right_pixel = pixels[row_start + col + 1]
                diff_bits = (diff_bits << 1) | (1 if left_pixel > right_pixel else 0)

        return diff_bits

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Computes Hamming distance (number of differing bits) between two 64-bit hashes."""
        return bin(hash1 ^ hash2).count("1")

    @classmethod
    def has_screen_changed(cls, frame1: Optional[ScreenFrame], frame2: Optional[ScreenFrame], threshold_bits: int = 3) -> bool:
        """Determines if the visual UI actually changed between two frames."""
        if frame1 is None or frame2 is None:
            return True
        dist = cls.hamming_distance(frame1.dhash, frame2.dhash)
        return dist > threshold_bits

    def capture(self, monitor_idx: int = 1) -> ScreenFrame:
        """Captures the primary monitor screen, downscales, and computes perceptual hash.

        Returns:
            ScreenFrame: Downscaled frame with coordinate transformation scalars.
        """
        now = time.time()
        sct = self._get_sct()

        # Handle headless or virtual environments
        if not sct:
            return self._generate_fallback_frame(now)

        try:
            # Monitor 1 is primary monitor; monitor 0 is all monitors combined
            monitors = sct.monitors
            target_mon = monitors[monitor_idx] if len(monitors) > monitor_idx else monitors[0]
            sct_img = sct.grab(target_mon)

            orig_w = sct_img.width
            orig_h = sct_img.height

            # Convert BGRA to RGB PIL Image
            raw_img = Image.frombytes("RGB", (orig_w, orig_h), sct_img.bgra, "raw", "BGRX")

            # Token-preserving downscaling (preserving aspect ratio)
            if max(orig_w, orig_h) > self.max_dimension:
                if orig_w >= orig_h:
                    new_w = self.max_dimension
                    new_h = max(1, int(orig_h * (self.max_dimension / orig_w)))
                else:
                    new_h = self.max_dimension
                    new_w = max(1, int(orig_w * (self.max_dimension / orig_h)))
                scaled_img = raw_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            else:
                scaled_img = raw_img
                new_w, new_h = orig_w, orig_h

            scale_x = orig_w / new_w
            scale_y = orig_h / new_h
            dhash_val = self.compute_dhash(scaled_img)

            return ScreenFrame(
                image=scaled_img,
                width=new_w,
                height=new_h,
                original_width=orig_w,
                original_height=orig_h,
                scale_x=scale_x,
                scale_y=scale_y,
                dhash=dhash_val,
                timestamp=now,
            )
        except Exception as exc:
            logger.warning("Screen capture error via mss: %s; attempting ImageGrab fallback.", exc)
            self._sct = None  # Reset stale GDI context
            try:
                from PIL import ImageGrab
                raw_img = ImageGrab.grab()
                if raw_img.mode != "RGB":
                    raw_img = raw_img.convert("RGB")
                orig_w, orig_h = raw_img.size

                if max(orig_w, orig_h) > self.max_dimension:
                    if orig_w >= orig_h:
                        new_w = self.max_dimension
                        new_h = max(1, int(orig_h * (self.max_dimension / orig_w)))
                    else:
                        new_h = self.max_dimension
                        new_w = max(1, int(orig_w * (self.max_dimension / orig_h)))
                    scaled_img = raw_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                else:
                    scaled_img = raw_img
                    new_w, new_h = orig_w, orig_h

                scale_x = orig_w / new_w
                scale_y = orig_h / new_h
                dhash_val = self.compute_dhash(scaled_img)
                return ScreenFrame(
                    image=scaled_img,
                    width=new_w,
                    height=new_h,
                    original_width=orig_w,
                    original_height=orig_h,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    dhash=dhash_val,
                    timestamp=now,
                )
            except Exception as exc2:
                logger.warning("ImageGrab fallback failed: %s; using fallback canvas.", exc2)
                return self._generate_fallback_frame(now)

    def _generate_fallback_frame(self, timestamp: float) -> ScreenFrame:
        """Generates a synthetic 1920x1080 canvas for headless / CI tests."""
        img = Image.new("RGB", (1920, 1080), color=(30, 30, 30))
        dhash_val = self.compute_dhash(img)
        return ScreenFrame(
            image=img,
            width=1920,
            height=1080,
            original_width=1920,
            original_height=1080,
            scale_x=1.0,
            scale_y=1.0,
            dhash=dhash_val,
            timestamp=timestamp,
        )
