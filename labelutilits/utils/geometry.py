import numpy as np

def distance(p1,p2):
    x0 = p1[0] - p2[0]
    y0 = p1[1] - p2[1]
    return np.sqrt(x0 ** 2 + y0 ** 2)

def perpendicular(x1, y1, x2, y2):
    xp, yp = rotate_matrix(x2, y2, np.pi / 2, x1, y1)
    return x1, y1, xp, yp

def rotate_matrix(x, y, angle, x_shift=0, y_shift=0, ):
    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * np.cos(angle)) - (y * np.sin(angle)) + x_shift
    yr = (x * np.sin(angle)) + (y * np.cos(angle)) + y_shift
    return xr, yr

def point_intersection(x1,y1,x2,y2, x3, y3, x4,y4):
    a = ((x1*y2 - y1*x2)*(x3-x4) - (x1 - x2)*(x3*y4 - y3*x4))/((x1 - x2)*(y3-y4) - (y1-y2)*(x3-x4))
    b = ((x1*y2 - y1*x2)*(y3-y4) - (y1 - y2)*(x3*y4 - y3*x4))/((x1 - x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return a,b

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
    x1 = x_center + a*np.cos(alpha)
    y1 = y_center + a*np.sin(alpha)  
    x2 = x_center - a*np.cos(alpha)
    y2 = y_center - a*np.sin(alpha)  
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
    x1 = x_center + b*np.cos(np.pi/2 + alpha)
    y1 = y_center + b*np.sin(np.pi/2 + alpha)  
    x2 = x_center - b*np.cos(np.pi/2 + alpha)
    y2 = y_center - b*np.sin(np.pi/2 + alpha)  
    return x1, y1, x2, y2

def correct_sequence(p1, p2, p3, p4):
    p_max_x, p_max_y, p_min_x, p_min_y = [None for i in range(4)]
    max_x = max_y = -10**10
    min_x = min_y = 10**10
    for p in (p1,p2,p3,p4):
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
    x1 = bx1 + a*np.cos(theta)
    y1 = by1 + a*np.sin(theta)
    x4 = bx1 - a*np.cos(theta)
    y4 = by1 - a*np.sin(theta)
    x2 = bx2 + a*np.cos(theta)
    y2 = by2 + a*np.sin(theta)
    x3 = bx2 - a*np.cos(theta)
    y3 = by2 - a*np.sin(theta)
    p1,p2,p3,p4 = correct_sequence((x1, y1),(x2, y2),(x3, y3),(x4, y4))
    return *p1, *p2, *p3, *p4