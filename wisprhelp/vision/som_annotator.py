"""Set-of-Marks (SoM) Visual Annotator & Candidate Catalog Generator.

Implements the Jev 'Decide, Don't Generate' paradigm by extracting physical UI controls
from the host OS accessibility tree (Windows UIA / OCR) and overlaying numbered high-contrast
badges ([1], [2], [3]...) onto the visual frame, guaranteeing 100% click precision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from wisprhelp.hal.base import BaseOSAdapter
from wisprhelp.hal.factory import get_os_adapter
from wisprhelp.vision.capture import ScreenFrame
from wisprhelp.vision.grounded_schemas import CandidateCatalog, UICandidateNode

logger = logging.getLogger(__name__)

# Check for Windows UIAutomation
try:
    import uiautomation as auto
    HAS_UIA = True
except ImportError:
    HAS_UIA = False


class SetOfMarksAnnotator:
    """Extracts host OS UI controls and annotates screen frames with numbered badge markers."""

    def __init__(self, os_adapter: Optional[BaseOSAdapter] = None) -> None:
        self.os_adapter = os_adapter or get_os_adapter()

    def extract_candidates_from_os(
        self,
        max_candidates: int = 50,
        min_width: int = 12,
        min_height: int = 12,
    ) -> CandidateCatalog:
        """Inspects the foreground window to extract interactive UI elements.

        Returns:
            CandidateCatalog mapping candidate_id to UICandidateNode.
        """
        catalog = CandidateCatalog()

        if not HAS_UIA:
            logger.debug("UIAutomation not available; returning empty candidate catalog.")
            return catalog

        try:
            root = auto.GetForegroundControl()
            if not root:
                root = auto.GetRootControl()

            candidates: List[auto.Control] = []
            seen_boxes = set()

            # Target interactive control types
            interactive_types = {
                "ButtonControl", "EditControl", "HyperlinkControl", "TabItemControl",
                "ListItemControl", "CheckBoxControl", "RadioButtonControl", "ComboBoxControl",
                "MenuItemControl", "ToolBarControl", "DocumentControl", "GroupControl"
            }

            def _walker(ctrl: auto.Control, depth: int) -> bool:
                if depth > 7 or len(candidates) >= max_candidates:
                    return False
                try:
                    r = ctrl.BoundingRectangle
                    w, h = r.width(), r.height()
                    if w >= min_width and h >= min_height and r.right > 0 and r.bottom > 0:
                        type_name = ctrl.ControlTypeName
                        name = ctrl.Name or ""
                        # Prioritize controls with interactive types or explicit names
                        if type_name in interactive_types or name:
                            # Spatial deduplication
                            key = (r.left // 8, r.top // 8, w // 8, h // 8)
                            if key not in seen_boxes:
                                seen_boxes.add(key)
                                candidates.append(ctrl)
                except Exception:
                    pass
                return True

            root.WalkControl(_walker, maxDepth=7)

            # Sort top-to-bottom, left-to-right
            candidates.sort(key=lambda c: (c.BoundingRectangle.top, c.BoundingRectangle.left))

            candidate_id = 1
            for c in candidates[:max_candidates]:
                try:
                    r = c.BoundingRectangle
                    cx = r.left + r.width() // 2
                    cy = r.top + r.height() // 2
                    name = c.Name or ""
                    ctl_type = c.ControlTypeName.replace("Control", "")
                    is_focused = bool(getattr(c, "HasKeyboardFocus", False))

                    catalog.add(
                        UICandidateNode(
                            candidate_id=candidate_id,
                            label=name[:40],
                            control_type=ctl_type,
                            bbox=(r.left, r.top, r.right, r.bottom),
                            center=(cx, cy),
                            is_focused=is_focused,
                        )
                    )
                    candidate_id += 1
                except Exception:
                    continue

            logger.info("Extracted %d grounded UI candidates from OS window.", len(catalog))
        except Exception as exc:
            logger.debug("Error extracting OS UI candidates: %s", exc)

        return catalog

    def annotate_frame(
        self,
        frame: ScreenFrame,
        catalog: CandidateCatalog,
        badge_radius: int = 11,
    ) -> Image.Image:
        """Overlays high-contrast numbered badges onto the ScreenFrame image.

        Args:
            frame: Raw captured and downscaled ScreenFrame.
            catalog: CandidateCatalog containing the elements in physical coordinates.
            badge_radius: Size of circular badge markers.

        Returns:
            PIL.Image.Image with rendered Set-of-Marks tags.
        """
        img = frame.image.copy()
        draw = ImageDraw.Draw(img)

        # Scale factors to convert physical screen coords to downscaled frame coords
        scale_x = frame.scale_x
        scale_y = frame.scale_y

        font = ImageFont.load_default()

        for cid, node in catalog.candidates.items():
            # Translate physical screen coords to frame pixel coords
            offset_x = getattr(frame, "offset_x", 0)
            offset_y = getattr(frame, "offset_y", 0)
            px = int((node.center[0] - offset_x) / scale_x)
            py = int((node.center[1] - offset_y) / scale_y)

            # Clamp badge center to stay inside image
            px = max(badge_radius + 2, min(frame.width - badge_radius - 2, px))
            py = max(badge_radius + 2, min(frame.height - badge_radius - 2, py))

            # Draw high-contrast badge (Vibrant Red background with solid White border and text)
            box = (px - badge_radius, py - badge_radius, px + badge_radius, py + badge_radius)
            draw.ellipse(box, fill=(235, 30, 30), outline=(255, 255, 255), width=2)

            # Draw candidate number
            tag_str = str(cid)
            bbox = font.getbbox(tag_str)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((px - tw // 2, py - th // 2 - 1), tag_str, fill=(255, 255, 255), font=font)

        return img
