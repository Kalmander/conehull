import numpy as np
from typing import Optional, List

def farthest_point(
        points: np.ndarray, 
        a: np.ndarray, 
        b: np.ndarray
) -> Optional[np.ndarray]:
    """
    Find the point in points that's farthest from the line ab.
    """
    ab = b - a
    abs_ab = np.linalg.norm(ab)
    max_dist = 0
    farthest_point = None

    for p in points:
        ap = p - a
        det = np.linalg.det(np.column_stack([ab, ap]))
        dist = abs(det) / abs_ab
        if dist >= max_dist: 
            max_dist = dist
            farthest_point = p

    return farthest_point

def points_on_left(
        points: np.ndarray, 
        a: np.ndarray, 
        b: np.ndarray
) -> List[np.ndarray]:
    """
        Find the points lying (strictly) left of the directed line ab.
    """
    ab = b - a
    left = []
    for p in points:
        ap = p - a
        det = np.linalg.det(np.column_stack([ab, ap]))
        if det > 1e-12:
            left.append(p)
    return left

def conehull_recurrence(
        points: np.ndarray, 
        a: np.ndarray, 
        b: np.ndarray
) -> List[np.ndarray]:
    """
        The recurrent step.
    """
    if len(points) == 0: return [a, b]
    farthest = farthest_point(points, a, b)

    # These are the points lying outside the triangle abp
    left1 = points_on_left(points, a, farthest)
    left2 = points_on_left(points, farthest, b)

    # Repeat on both sides of the triangle
    hull1 = conehull_recurrence(left1, a, farthest)
    hull2 = conehull_recurrence(left2, farthest, b)

    return hull1[:-1] + hull2 # Remove duplicate farthest point

def conehull(points: np.ndarray) -> np.ndarray:
    """
        Compute the convex hull of a set of points
        with respect to a cone, using an algorithm
        based on QuickHull.
    """
    if len(points) < 3:
        return points
    points = np.array(points)
    leftmost = points[np.argmin(points[:, 0])]
    rightmost = points[np.argmax(points[:, 0])]

    above = points_on_left(points, leftmost, rightmost)
    below = points_on_left(points, rightmost, leftmost)

    upper = conehull_recurrence(above, leftmost, rightmost)
    lower = conehull_recurrence(below, rightmost, leftmost)

    hull = upper[:-1] + lower[:-1] # Remove duplicate endpoints

    # Remove duplicates in case of collinear points
    unique_hull = []
    for p in hull:
        if not any(np.allclose(p, q) for q in unique_hull):
            unique_hull.append(p)

    return np.array(unique_hull)
