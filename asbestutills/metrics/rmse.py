import numpy as np
from asbestutills.utils.geometry import coords_max_line, max_distance
from sklearn.neighbors import BallTree
from asbestutills.metrics.map import predictionformat

class Polygone:
    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy
        # self._centr = None
        self._validate_coords()
        self.__calculate()
        
    def _validate_coords(self):
        if len(self.xx) < 3:
            raise ValueError("Полигон должен содержать минимум 3 точки")
        if len(self.xx) != len(self.yy):
            raise ValueError("Размеры координат не совпадают")
        
    def __calculate(self):
        xc = np.sum(self.xx)/len(self.xx)
        yc = np.sum(self.yy)/len(self.yy)
        self._centr = (xc,yc)
        x1, y1, x2, y2 = coords_max_line(self.xx, self.yy)
        self._max_size = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    @property
    def centr(self):
        return self._centr
    
    @property
    def max_size(self):
        return self._max_size


class PolygonManager:
    def __init__(self, polygones, width, height):
        self.width = width
        self.height = height
        self.polygones = [Polygone(s[::2],s[1::2]) for s in polygones]
        self._centrs = [x.centr for x in self.polygones]
        self._tree = BallTree(self._centrs, leaf_size=2)

    @property
    def centrs(self):
        return self._centrs

    def neighbor_polygone(self, q):
        """
        return:
        tuple: (index, center, distance)
            - index : int
            - center : array-like
            - distance : float
        Example:
        -------
        >>> neighbor_polygone([5, 10])
        (3, array([4.5, 9.2]), 1.2)
        """
        q = np.array(q).reshape(1,-1)
        dist, ind = self._tree.query(q, k=1)
        return ind[0][0], self.polygones[ind[0][0]].centr, dist

    def get_maxsize(self, k_indx):
        return self.polygones[k_indx].max_size
    
def point_in_polygon(p, polygon):
    x, y = p
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def collect_sizes(segments, obb_pred, width, height):
    """
    segments:
        [
            [x1, y1, x2, y2, ..., xn, yn],  # Координаты точек первого полигона
            [x1, y1, x2, y2, ..., xk, yk],  # Координаты точек второго полигона
        ...
        ]
    obb_pred: np.array предсказания обрамляющими рамками без p класса
        для сегментации сначало координаты бокса x,y,w,h, потом segment. [:,4:]

    """
    manager = PolygonManager(segments, width, height)
    pred_format = predictionformat(obb_pred)

    #центр масс фрагмента
    if pred_format == 'obb':
        obb_pred = np.array(obb_pred)[:,1:]
        obb_pred[:,::2]*=width
        obb_pred[:,1::2]*=height
        xc = np.sum(obb_pred[:,::2],axis=1)/4
        yc = np.sum(obb_pred[:,1::2],axis=1)/4
        yc = yc.astype(np.int32)
        xc = xc.astype(np.int32)
    else:
        xc = []
        yc = []
        for segment in obb_pred:
            segment = np.array(segment)
            l = len(segment[1::2])
            cx = segment[1::2].sum()/l
            cy = segment[2::2].sum()/l
            xc.append(cx)
            yc.append(cy)

        yc = np.array(yc)
         #коррекция
        xc = np.array(xc)*width
        yc = np.array(yc)*height
        
        xc = xc.astype(np.int32)
        yc = yc.astype(np.int32)
    seg_maxsize = []
    bbox_sizes =  [] #размеры obb

    for i in range(len(xc)):
        k = manager.neighbor_polygone((xc[i], yc[i]))[0]
        pol = manager.polygones[k]
        # if pred_format == 'obb':
        r=np.array([(x,y) for x,y in zip(pol.xx,pol.yy)]).astype(np.int32)        
        # print(xc[i], yc[i],r)
        p_in = point_in_polygon((xc[i], yc[i]),r)
        if not p_in:
            continue
        else:
            seg_maxsize.append(pol.max_size)
            if pred_format == 'obb':
                coords = obb_pred[i]
                dx = np.sqrt((coords[0] - coords[2]) ** 2 + (coords[1] - coords[3]) ** 2)
                dy = np.sqrt((coords[2] - coords[4]) ** 2 + (coords[3] - coords[5]) ** 2)
                d=max(dx, dy)
            else:#контур
                coords = obb_pred[i][1:]
                d = max_distance(coords[0::2]*width,coords[1::2]*height,)

            bbox_sizes.append(d)
             
    return seg_maxsize, bbox_sizes

    # mean_absolute_error(seg_maxsize,bbox_sizes)