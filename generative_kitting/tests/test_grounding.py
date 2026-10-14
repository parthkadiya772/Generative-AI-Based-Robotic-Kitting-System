"""Grasp-point estimate from depth samples (orchestration.workflow_engine._top_surface_point)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.workflow_engine import _top_surface_point

FLOOR, TOP = 0.795, 0.835          # bin floor and large_gear top, measured 2026-10-14


def _gear_grid(cx=-1.52, cy=-0.72, n=9, half=0.05, bore=True):
    """n x n samples over a 95 mm gear on the bin floor, with a bore at the centre."""
    pts = []
    for i in range(n):
        for j in range(n):
            x = cx - half + 2 * half * i / (n - 1)
            y = cy - half + 2 * half * j / (n - 1)
            r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            on_gear = r <= 0.0475 and not (bore and r < 0.012)
            pts.append({"x": x, "y": y, "z": TOP if on_gear else FLOOR})
    return pts


def test_top_surface_ignores_bore_and_floor():
    x, y, z, n = _top_surface_point(_gear_grid())
    assert z == pytest.approx(TOP)
    assert (x, y) == pytest.approx((-1.52, -0.72), abs=0.002)
    assert n > 0


def test_box_shifted_off_the_part_still_finds_its_top():
    # A box 30 mm off: XY moves toward the overlap, Z stays the top.
    pts = [p for p in _gear_grid(cx=-1.49) if p["x"] < -1.47]
    x, y, z, _ = _top_surface_point(pts)
    assert z == pytest.approx(TOP)


def test_errors_and_empty_input():
    assert _top_surface_point([]) is None
    assert _top_surface_point([{"error": "no valid depth at pixel"}]) is None
    assert _top_surface_point([{"x": 0.0, "y": 0.0, "z": 1.0}, {"error": "x"}])[2] == 1.0
