import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from matplotlib import patches


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Node:
    def __init__(self, x0, y0, w, h, points):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.points = points
        self.children = []


def recursive_to_quad(node, threshold):
    """
    Recursively split the node into 4 quads. top-left, top-right, bottom-left and bottom-right of the center point

    Args:
        node: QTree Node to be split if its data point count is higher than the threshold
        threshold: Data point count threshold
    """
    if len(node.points) <= threshold:
        return

    w_ = float(node.width / 2)
    h_ = float(node.height / 2)
    c = Point(node.x0 + w_, node.y0 + h_)

    bottom_left = []
    bottom_right = []
    top_left = []
    top_right = []

    for p in node.points:
        if p.x < c.x:
            if p.y < c.y:
                bottom_left.append(p)
            else:
                top_left.append(p)
        else:
            if p.y < c.y:
                bottom_right.append(p)
            else:
                top_right.append(p)

    node.points = None
    x1 = Node(node.x0, node.y0, w_, h_, bottom_left)
    recursive_to_quad(x1, threshold)

    x2 = Node(node.x0, node.y0 + h_, w_, h_, top_left)
    recursive_to_quad(x2, threshold)

    x3 = Node(node.x0 + w_, node.y0, w_, h_, bottom_right)
    recursive_to_quad(x3, threshold)

    x4 = Node(node.x0 + w_, node.y0 + h_, w_, h_, top_right)
    recursive_to_quad(x4, threshold)

    node.children = [x1, x2, x3, x4]


def contains(x, y, w, h, points):
    pts = []
    for point in points:
        if x <= point.x <= x + w and y <= point.y <= y + h:
            pts.append(point)
    return pts


def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children


class QTree:
    def __init__(self, k, n=None, points=None):
        self.threshold = k
        self.points = points if points else [Point(random.normal(0, 2), random.normal(0, 2)) for x in range(n)]
        min_x = min([p.x for p in self.points])
        min_y = min([p.y for p in self.points])
        max_x = max([p.x for p in self.points])
        max_y = max([p.y for p in self.points])
        self.root = Node(min_x, min_y, max_x-min_x, max_y-min_y, self.points)
        recursive_to_quad(self.root, self.threshold)

    def aggregate_nodes(self):
        pass

    def range_query(self, center, radius):
        pass

    def graph(self):
        fig = plt.figure(figsize=(12, 8))
        plt.title("Quadtree")
        ax = fig.add_subplot(111)
        c = find_children(self.root)
        print(f'Number of segments: {len(c)}')
        areas = set()
        for el in c:
            areas.add(el.width * el.height)
        print(f'Minimum segment area: {min(areas)} units')

        # plot all points
        for n in c:
            ax.add_patch(patches.Rectangle((n.x0, n.y0), n.width, n.height, fill=False, color='gray', linewidth=0.2))
        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        plt.plot(x, y, linestyle='', color='gray', marker='.', markersize=1)

        # plot query
        query_point = np.array([2, 2])
        query_radius = 2
        ax.add_artist(plt.Circle(tuple(query_point), query_radius, color='#FF000033'))
        within_circle = [p for p in self.points if np.linalg.norm(np.array([p.x, p.y])-query_point) < query_radius]
        x = [point.x for point in within_circle]
        y = [point.y for point in within_circle]
        plt.plot(x, y, 'r.', markersize=3)
        plt.show()


if __name__ == '__main__':
    qt = QTree(10, 2000)
    qt.graph()
    print('Done')
