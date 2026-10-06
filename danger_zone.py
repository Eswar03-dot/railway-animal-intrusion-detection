import cv2
import numpy as np


# Railway danger-zone polygon
DANGER_ZONE = np.array([
    (481, 604),
    (532, 601),
    (603, 606),
    (676, 608),
    (731, 620),
    (792, 620),
    (912, 633),
    (968, 640),
    (1097, 652),
    (1188, 662),
    (1248, 668),
    (1281, 672),
    (1347, 681),
    (1371, 689),
    (1492, 739),
    (1577, 747),
    (1675, 782),
    (1724, 816),
    (1694, 908),
    (1675, 956),
    (1672, 1028),
    (1628, 1065),
    (1414, 1059),
    (1316, 1065),
    (1176, 1065),
    (1006, 1059),
    (832, 1065),
    (670, 1062),
    (491, 1075),
    (393, 1073),
    (283, 1051),
    (120, 1048),
    (51, 1030),
    (27, 934)
], dtype=np.int32)


def is_inside_danger_zone(x, y):
    """
    Check whether a point is inside the railway danger zone.

    Returns:
        True  -> point is inside
        False -> point is outside
    """

    point = (int(x), int(y))

    result = cv2.pointPolygonTest(
        DANGER_ZONE,
        point,
        False
    )

    return result >= 0