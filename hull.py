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
        Find the points lying left of the directed line ab.
    """
    ab = b - a
    left = []
    for p in points:
        ap = p - a
        det = np.linalg.det(np.column_stack([ab, ap]))
        if det > 1e-12:
            left.append(p)
    return left

