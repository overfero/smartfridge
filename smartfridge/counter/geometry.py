"""Geometry primitives for line-crossing detection."""

from __future__ import annotations

Point = tuple[int | float, int | float]


def _ccw(a: Point, b: Point, c: Point) -> bool:
    """Return True if A→B→C make a counter-clockwise turn."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return True if segment AB intersects segment CD."""
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def is_point_below_line(point: Point, line_start: Point, line_end: Point) -> bool:
    """Return True if point is below the line (higher y value in image coords)."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    if x2 == x1:
        return False
    line_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return y > line_y


def is_point_above_line(point: Point, line_start: Point, line_end: Point) -> bool:
    """Return True if point is above the line (lower y value in image coords)."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    if x2 == x1:
        return False
    line_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return y < line_y


def get_direction(p1: Point, p2: Point) -> str:
    """Return cardinal direction string (e.g. 'NorthEast') from p1 → p2."""
    vertical   = "South" if p1[1] > p2[1] else ("North" if p1[1] < p2[1] else "")
    horizontal = "East"  if p1[0] > p2[0] else ("West"  if p1[0] < p2[0] else "")
    return vertical + horizontal
