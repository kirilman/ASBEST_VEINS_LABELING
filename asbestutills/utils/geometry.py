import numpy as np
from typing import Tuple


def distance(p1, p2):
    x0 = p1[0] - p2[0]
    y0 = p1[1] - p2[1]
    return np.sqrt(x0**2 + y0**2)


def perpendicular(x1, y1, x2, y2):
    xp, yp = rotate_matrix(x2, y2, np.pi / 2, x1, y1)
    return x1, y1, xp, yp


def rotate_matrix(
    x,
    y,
    angle,
    x_shift=0,
    y_shift=0,
):
    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * np.cos(angle)) - (y * np.sin(angle)) + x_shift
    yr = (x * np.sin(angle)) + (y * np.cos(angle)) + y_shift
    return xr, yr


def point_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    a = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    b = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    return a, b


def coords_main_line(x_center, y_center, a, alpha):
    """
        Get the coordinates x1,y1,x2,y2 for the major axis of an ellipse with a - major radius
    Args:
        x_center (_type_): _description_
        y_center (_type_): _description_
        b (_type_): _description_
        alpha (_type_): _description_

    Returns:
        x1, y1, x2, y2 : coordinates major axes
    """
    x1 = x_center + a * np.cos(alpha)
    y1 = y_center + a * np.sin(alpha)
    x2 = x_center - a * np.cos(alpha)
    y2 = y_center - a * np.sin(alpha)
    return x1, y1, x2, y2


def coords_other_line(x_center, y_center, b, alpha):
    """
        Get the coordinates x1,y1,x2,y2 for the second axis of an ellipse with b - second radius
    Args:
        x_center (_type_): _description_
        y_center (_type_): _description_
        b (_type_): _description_
        alpha (_type_): _description_

    Returns:
        x1, y1, x2, y2 : coordinates major axes
    """
    x1 = x_center + b * np.cos(np.pi / 2 + alpha)
    y1 = y_center + b * np.sin(np.pi / 2 + alpha)
    x2 = x_center - b * np.cos(np.pi / 2 + alpha)
    y2 = y_center - b * np.sin(np.pi / 2 + alpha)
    return x1, y1, x2, y2


def correct_sequence(p1, p2, p3, p4):
    p_max_x, p_max_y, p_min_x, p_min_y = [None for i in range(4)]
    max_x = max_y = -(10**10)
    min_x = min_y = 10**10
    for p in (p1, p2, p3, p4):
        if p[0] > max_x:
            max_x = p[0]
            p_max_x = p
        if p[0] < min_x:
            min_x = p[0]
            p_min_x = p
        if p[1] > max_y:
            max_y = p[1]
            p_max_y = p
        if p[1] < min_y:
            min_y = p[1]
            p_min_y = p
    return p_min_y, p_max_x, p_max_y, p_min_x


def coords_obb(bx1, by1, bx2, by2, a, theta):
    """
        Get obb coordinates x1, y1, x2, y2, x3, y3, x4, y4 using coordinates second axes line of ellipse
        and ellipse parameters: a,theta
    Args:
        bx1 (_type_): _description_
        by1 (_type_): _description_
        bx2 (_type_): _description_
        by2 (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    x1 = bx1 + a * np.cos(theta)
    y1 = by1 + a * np.sin(theta)
    x4 = bx1 - a * np.cos(theta)
    y4 = by1 - a * np.sin(theta)
    x2 = bx2 + a * np.cos(theta)
    y2 = by2 + a * np.sin(theta)
    x3 = bx2 - a * np.cos(theta)
    y3 = by2 - a * np.sin(theta)
    p1, p2, p3, p4 = correct_sequence((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    return *p1, *p2, *p3, *p4


def coords_max_line(x_coords, y_coords):
    """
        Get coords max_line x1, y1, x2, y2
    Args:
        x_coords (np.darray): x coords polygone
        y_coords (np.darray): y coords polygone
    """
    max_distance = 0
    points = [(x, y) for x, y in zip(x_coords, y_coords)]
    main_point = None

    for p1 in points:
        for p2 in points:
            dist = max(max_distance, distance(p1, p2))
            if dist > max_distance:
                max_distance = dist
                main_point = (p1, p2)
    x1, y1 = main_point[0]
    x2, y2 = main_point[1]
    return x1, y1, x2, y2


def position(x, y, x1, y1, x2, y2):
    """point position right or left
    Args:
        x (_type_): point
        y (_type_): point
        x1 (_type_): line
        y1 (_type_): line
        x2 (_type_): line
        y2 (_type_): line

    Returns:
        bool:
    """
    return (y2 - y1) * (x - x1) - (y - y1) * (x2 - x1)


def distance_to_perpendicular(a, b, c, x, y):
    return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)


def line_from_points(p1, p2):
    a = p2[1] - p1[1]
    b = p2[0] - p1[0]
    c = -a * (p1[0]) + b * (p1[1])
    return a, -b, c


def vec_from_points(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return a, b


def coords_perpendicular(p1: Tuple, p2: Tuple, p3: Tuple):
    """
    Not worked!!!
    Args:
        p1 (Tuple): main point
        p2 (Tuple): first point of line
        p3 (Tuple): second point of line
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4 = (
        (x2 - x1) * (y2 - y1) * (y3 - y1) + x1 * pow(y2 - y1, 2) + x3 * pow(x2 - x1, 2)
    ) / (pow(y2 - y1, 2) + pow(x2 - x1, 2))
    y4 = (y2 - y1) * (x4 - x1) / (x2 - x1) + y1


def dot_product_angle(v1, v2):
    if isinstance(v1, list):
        v1 = np.array(v1)
    if isinstance(v2, list):
        v2 = np.array(v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        res = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return res
    return 0


def coords_other_line_by_coords(x, y):
    """
    Get coords second line of polygones x, y
    Returns:
        x1, y1, x2, y2 coords of line
    """
    x1, y1, x2, y2 = coords_max_line(x, y)
    a, b, c = line_from_points((x1, y1), (x2, y2))
    x_c1, y_c1 = x[0], y[0]
    x_c2, y_c2 = x_c1, y_c1
    ro_max_right, ro_max_left = 0, 0
    for xi, yi in zip(x, y):
        d = distance_to_perpendicular(a, b, c, xi, yi)
        if position(xi, yi, x1, y1, x2, y2) > 0:
            if d > ro_max_right:
                ro_max_right = d
                x_c1, y_c1 = xi, yi
        else:
            if d > ro_max_left:
                ro_max_left = d
                x_c2, y_c2 = xi, yi

    return x_c1, y_c1, x_c2, y_c2


def segment2obb(x_coords, y_coords):
    """
    Get oriented box coordinates from segmentation mask coordinates (x_coords, y_coords)
    Returns:
    x0, y0, x1, y1, x2, y2, x3, y3
    """
    ax1, ay1, ax2, ay2 = coords_max_line(x_coords, y_coords)
    if ax2 > ax1 and ay2 > ay1:
        ax2, ax1 = ax1, ax2
        ay2, ay1 = ay1, ay2
    Points = [(x, y) for x, y in zip(x_coords, y_coords)]
    max_dist_left = 0
    max_dist_right = 0
    for point in Points:
        if position(point[0], point[1], ax1, ay1, ax2, ay2) > 0:
            A, B, C = line_from_points((ax1, ay1), (ax2, ay2))
            d = distance_to_perpendicular(A, B, C, point[0], point[1])
            if abs(d) > max_dist_right:
                max_dist_right = d
                bx2, by2 = point
        else:
            A, B, C = line_from_points((ax1, ay1), (ax2, ay2))
            d = distance_to_perpendicular(A, B, C, point[0], point[1])
            if abs(d) > max_dist_left:
                max_dist_left = d
                bx1, by1 = point

    px1, px2 = point_intersection(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    n1, n2 = vec_from_points((px1, px2), (ax1, ay1))
    m1, m2 = vec_from_points((px1, px2), (px1 + 1000, px2))
    theta = dot_product_angle([n1, n2], [m1, m2])

    A, B, C = line_from_points((ax1, ay1), (ax2, ay2))
    h2 = distance_to_perpendicular(A, B, C, bx2, by2)
    h1 = distance_to_perpendicular(A, B, C, bx1, by1)
    alpha = np.pi / 2 - theta
    if ay1 < px2:
        alpha = np.pi / 2 + theta
    dy = h1 * np.sin(alpha)
    dx = h1 * np.cos(alpha)
    # coords obb obx1, oby1, ...
    obx1 = ax1 + dx
    oby1 = ay1 - dy
    obx4 = ax2 + dx
    oby4 = ay2 - dy
    dy = h2 * np.sin(alpha)
    dx = h2 * np.cos(alpha)
    obx2 = ax1 - dx
    oby2 = ay1 + dy
    obx3 = ax2 - dx
    oby3 = ay2 + dy
    op1, op2, op3, op4 = correct_sequence(
        (obx1, oby1), (obx2, oby2), (obx3, oby3), (obx4, oby4)
    )
    return *op1, *op2, *op3, *op4


def maxdistance_beetwen_points(xx, yy):
    p1, p2 = (xx[0], yy[0]), (xx[1], yy[1])
    max_ro = -(10**6)
    for k, (x1, y1, x2, y2) in enumerate(zip(xx[:-1], yy[:-1], xx[1:], yy[1:])):
        d = distance((x1, y1), (x2, y2))

        if d > max_ro:
            max_ro = d
            p1 = (x1, y1)
            p2 = (x2, y2)
            index = k
    return max_ro, p1, p2, index


def add_points_to_polygone(xx, yy, N):
    start_len = len(xx)
    for i in range(0, N):
        ro, p1, p2, index = maxdistance_beetwen_points(xx, yy)
        new_x = p1[0] + (p2[0] - p1[0]) / 2
        new_y = p1[1] + (p2[1] - p1[1]) / 2
        xx = np.hstack((xx[: index + 1], [new_x], xx[index + 1 :]))  #!
        yy = np.hstack((yy[: index + 1], [new_y], yy[index + 1 :]))  #!
    return xx, yy
