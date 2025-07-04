import numpy as np

def farthest_point(points, a, b):
    """
    Find the point in points that's farthest from the line ab.

    Args:
        points (np.array): (number_of_points, n) numpy array
        a (np.array): (n,) numpy array
        b (np.array): (n,) numpy array

    Returns:
        np.array: (n,) numpy array representing the farthest point
                       (or None if empty)
    """
    ab = b - a
    abs_ab = np.linalg.norm(ab)
    max_dist = 0
    farthest_point = None

    for p in points:
        ap = p - a
        det = np.linalg.det(np.column_stack([ap, ab]))
        dist = det / abs_ab
        if dist >= max_dist: 
            max_dist = dist
            farthest_point = p

    return farthest_point
