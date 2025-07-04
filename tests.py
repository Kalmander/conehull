import numpy as np
from hull import *

res = farthest_point(
    np.array([
        [1, 1],
        [2, 1],
        [2, 3],
        [1, 5],
        [1, 7],
    ]),
    np.array([0,0]),
    np.array([0,4])
)

res = points_on_left(
    np.array([
        [1, 1],
        [2, 1],
        [2, 3],
        [1, 5],
        [1, 7],
    ]),
    np.array([0,7]),
    np.array([0,4])
)


points = np.array([
    [0, 0], [1, 1], [2, 0], [1, -1], 
    [0, -2], [-1, -1], [-2, 0], [-1, 1]
])
hull = conehull(points)

print(hull)
