from collections import namedtuple

from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

LinregressResult = namedtuple('LinregressResult', ('slope', 'intercept',
                                                   'rvalue', 'pvalue',
                                                   'stderr'))
Point = namedtuple("Point", ("x", "y"))
Line = namedtuple("Line", ("p1", "p2"))

def point_in_polygon(target, poly):
    """ Ray-casting point-in-polygon algorithm (`http://en.wikipedia.org/wiki/Point_in_polygon`).

    Args:
        target (array-like): point to test
        poly (ndarray): shape of (n, 2), vertexes comprising the polygon.
    
    Returns:
        bool: if target is inside the poly
    """
    target = Point(*target)

    inside = False
    # Build list of coordinate pairs
    # First, turn it into named tuples

    poly = list(map(lambda p: Point(*p), poly))

    # Make two lists, with list2 shifted forward by one and wrapped around
    list1 = poly
    list2 = poly[1:] + [poly[0]]
    poly = list(map(Line, list1, list2))

    for l in poly:
        p1 = l.p1
        p2 = l.p2

        if p1.y == p2.y:
            # This line is horizontal and thus not relevant.
            continue
        if max(p1.y, p2.y) < target.y or target.y <= min(p1.y, p2.y):
            # This line is too high or low
            continue
        if target.x < min(p1.x, p2.x):
            # Ignore this line because it's to the right of our point
            continue
        # Now, the line still might be to the right of our target point, but 
        # still to the right of one of the line endpoints.
        rise = p1.y - p2.y
        run =  p1.x - p2.x
        try:
            slope = rise / float(run)
        except ZeroDivisionError:
            slope = float('inf')

        # Find the x-intercept, that is, the place where the line we are
        # testing equals the y value of our target point.

        # Pick one of the line points, and figure out what the run between it
        # and the target point is.
        rise_to_intercept = target.y - p1.y
        x_intercept = p1.x + rise_to_intercept / slope
        if target.x < x_intercept:
            # We almost crossed the line.
            continue

        inside = not inside

    return inside


def split_rectangle(lines, shape):
    """Split with lines a rectangle locating at origin. 
    
    Args:
        lines (List or ndarray): lines represented by points
        shape (tuple or List): shape of rectangle to split
    
    Returns:
        (List[ndarray]): splitted polygons represented by vertexes
    """
    w, h = shape
    y_min = 0
    lrs = [linregress(line) for line in lines]
    if len(lrs) == 1:
        y_min = lines[0][:, 1].min()
    else:
        for lr1, lr2 in zip(lrs[:-1], lrs[1:]):
            x3 = (lr1.intercept - lr2.intercept) / (lr2.slope - lr1.slope)
            y_min = max(y_min, x3 * lr1.slope + lr1.intercept)
            
    lrs.insert(0, LinregressResult(-0.001, y_min, 0, 0, 0))
    lrs.insert(len(lrs), LinregressResult(0.001, y_min, 0, 0, 0))
    
    lanes = []
    for lr1, lr2 in zip(lrs[:-1], lrs[1:]):
        polygons = []
        x1 = np.clip((h - lr1.intercept) / lr1.slope, 0, w)
        y1 = x1 * lr1.slope + lr1.intercept
        polygons.append([x1, y1])

        x2 = np.clip((h - lr2.intercept) / lr2.slope, 0, w)
        y2 = x2 * lr2.slope + lr2.intercept

        if x1 == 0 and x2 > 0:
            polygons.append([0, h])
        if x1 < w and x2 == w:
            polygons.append([w, h])

        polygons.append([x2, y2])

        x3 = (lr1.intercept - lr2.intercept) / (lr2.slope - lr1.slope)
        y3 = np.clip(x3 * lr1.slope + lr1.intercept, y_min, h)
        if y3 == y_min and x3 != (y_min - lr2.intercept ) / lr2.slope:
            polygons.append([(y_min - lr2.intercept ) / lr2.slope, y_min])
            polygons.append([(y_min - lr1.intercept ) / lr1.slope, y_min])
        else:
            polygons.append([x3, y3])

        lane = np.vstack(polygons)

        lanes.append(lane)

    return lanes


if __name__ == '__main__':
    w = h = 10.
    ls = [(-2, 9), (10, -35), (1, -4)] # List[(slope, intercept)]
    lines = []
    xs = [np.linspace(0, 2, 18), np.linspace(4, 4.5, 18), np.linspace(8, 10, 18)]
    # test cases
    # ls = []; xs = []
    # ls = ls[:2]; xs = xs[:2]
    # ls = ls[:1]; xs = xs[:1]
    # ls = ls[1:2]; xs = xs[1:2]
    # ls = ls[2:3]; xs = xs[2:3]
    for i, l in enumerate(ls):
        y = xs[i] * l[0] + l[1]
        lines.append(np.stack([xs[i], y], -1))

    for line in lines:
        plt.scatter(line[:, 0], -line[:, 1])

    lanes = split_rectangle(lines, (w, h))
    main_lane = [point_in_polygon([w / 2, h], _) for _ in lanes].index(True)

    for i, lane in enumerate(lanes):
        lane = np.vstack([lane, lane[None, 0]])
        plt.plot(lane[:, 0], -lane[:, 1])
    plt.arrow(w / 2, -h, dx=0., dy=1, width=0.2, ec='blue')
    plt.text(lanes[main_lane].mean(0)[0], -lanes[main_lane].mean(0)[1], s='Main')
    
    # point-in-lane test
    for p in np.random.uniform(0, w, (250, 2)):
        inside = point_in_polygon(p, lanes[main_lane])
        color = 'blue' if inside else 'black'
        marker = '+' if inside else 'x'
        plt.scatter(p[0], -p[1], marker=marker, color=color)

    plt.xlim(- 0.1, w + 0.1)
    plt.ylim(- h - 0.1,  0.1)
    plt.axis('off')
    
