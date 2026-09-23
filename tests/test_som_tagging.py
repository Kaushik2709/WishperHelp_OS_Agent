"""Unit Tests for Set-of-Marks (SoM) Visual Annotator & Candidate Catalog.

Verifies element cataloging, candidate selection constraints, and visual badge rendering.
"""

from unittest.mock import MagicMock
from PIL import Image
import pytest

from wisprhelp.vision.capture import ScreenFrame
from wisprhelp.vision.grounded_schemas import CandidateCatalog, UICandidateNode
from wisprhelp.vision.som_annotator import SetOfMarksAnnotator


def test_candidate_catalog_add_and_lookup():
    catalog = CandidateCatalog()
    node = UICandidateNode(
        candidate_id=1,
        label="Search",
        control_type="Button",
        bbox=(100, 200, 180, 240),
        center=(140, 220),
        is_focused=True,
    )
    catalog.add(node)

    assert len(catalog) == 1
    assert catalog.get(1) == node
    assert catalog.get(99) is None
    assert node.width == 80
    assert node.height == 40


def test_candidate_catalog_prompt_formatting():
    catalog = CandidateCatalog()
    catalog.add(
        UICandidateNode(
            candidate_id=1,
            label="Submit",
            control_type="Button",
            bbox=(10, 20, 50, 40),
            center=(30, 30),
            is_focused=False,
        )
    )
    catalog.add(
        UICandidateNode(
            candidate_id=2,
            label="Email Input",
            control_type="Edit",
            bbox=(60, 20, 200, 40),
            center=(130, 30),
            is_focused=True,
        )
    )

    context = catalog.to_prompt_context()
    assert "[1] Button: 'Submit'" in context
    assert "[2] Edit: 'Email Input' [FOCUSED]" in context


def test_som_badge_rendering():
    annotator = SetOfMarksAnnotator()
    # Create test image frame
    base_img = Image.new("RGB", (640, 480), color=(50, 50, 50))
    frame = ScreenFrame(
        image=base_img,
        width=640,
        height=480,
        original_width=1920,
        original_height=1080,
        scale_x=3.0,
        scale_y=2.25,
        dhash=12345,
        timestamp=100.0,
    )

    catalog = CandidateCatalog()
    catalog.add(
        UICandidateNode(
            candidate_id=1,
            label="Click Me",
            control_type="Button",
            bbox=(300, 300, 450, 350),
            center=(375, 325),
        )
    )

    tagged_img = annotator.annotate_frame(frame, catalog, badge_radius=10)
    assert tagged_img is not None
    assert tagged_img.size == (640, 480)

    # Pixel at badge center should be the bright red badge fill (235, 30, 30) or white border
    bx = int((375 - getattr(frame, "offset_x", 0)) / frame.scale_x)
    by = int((325 - getattr(frame, "offset_y", 0)) / frame.scale_y)
    r, g, b = tagged_img.getpixel((bx, by))
    # Red component should be high
    assert r > 200
