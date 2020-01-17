import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from matplotlib import patches
from bitarray import bitarray
import struct


class Node:
    def __init__(self, df, dimension_cols, data_point_indexes=None, widths=None, threshold=100):
        """
        Store the row indexes in df balanced based on the center point and coordinate_cols of those rows.
        Each node has a N-dimensional box assigned to it which has upper and lower bound with equal distance
        to the center point (Width).

        Args:
            df(DataFrame): The DataFrame containing the coordinate columns and the data.
            dimension_cols(List): The list of coordinate columns in df parameter.
            data_point_indexes(List): Indexes of rows in df that must be assign to this Node.
                All rows will be considered if None is passed in.
            widths(List): The distance between the center point and upper and lower bound.
                Will be calculated based on the minimum and maximum values of each dimension column it it is None.
            threshold: Maximum number of points that each leaf node can store
        """
        if not data_point_indexes:
            data_point_indexes = df.index.to_list()

        minimums = df[dimension_cols].iloc[data_point_indexes].min()
        maximums = df[dimension_cols].iloc[data_point_indexes].max()

        self.center = [(l + u) / 2 for (l, u) in zip(minimums, maximums)]
        """The coordinates of the center point"""

        self.widths = widths if widths else [(u - l) / 2 for (l, u) in zip(minimums, maximums)]
        """The distances from center"""

        self.threshold = threshold
        """Maximum number of points that each leaf node can store"""

        self.df = df
        """The DataFrame storing the data including the coordinates columns"""

        self.coordinate_cols = dimension_cols
        """List of columns defining the dimensions for each data point coordination in order"""

        if len(data_point_indexes) <= threshold:
            self.indexes = data_point_indexes
        else:
            self.children = [None] * (2 ** len(self.center))
            """Contains 2 ^ D children"""

            segregated_indexes = []
            for _ in range(2 ** len(self.center)):
                segregated_indexes.append(list())

            for idx in data_point_indexes:
                segregated_indexes[self._get_child_idx(idx)].append(idx)

            # TODO: Use decimal for more accuracy
            child_widths = [w / 2 for w in self.widths]
            for i in range(len(segregated_indexes)):
                self.children[i] = Node(df, dimension_cols, segregated_indexes[i], child_widths, threshold)

    def _get_child_idx(self, data_point_idx: int) -> int:
        """
        Return the index of the child which provided data_point belongs to.
        This function is the core logic of QTree data structure

        Args:
            data_point_idx: The index of the data point in self.df

        Returns:
             int: The index of the chile that contains (or must) the provided data_point_idx

        """
        # TODO: Think of a way to calculate the child indexes as a batch instead of one by one.
        #       For example address coordination columns in df

        child_index = 0
        for i in range(len(self.coordinate_cols)):
            child_index = (child_index << 1) | int(
                self.center[i] <= self.df[self.coordinate_cols[i]].iloc[data_point_idx])
        return child_index


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
    def __init__(self, df, coordinate_cols, leaf_points_limit=100):
        """

        Args:
            df(DataFrame): the DataFrame containing the coordinate columns and the data
            coordinate_cols(List): the list of coordinate column names in df parameter
            leaf_points_limit: maximum number of points that each leaf node can store
        """
        self.threshold = leaf_points_limit
        self.df = df

        self.root = Node(df, coordinate_cols)

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
        within_circle = [p for p in self.points if np.linalg.norm(np.array([p.x, p.y]) - query_point) < query_radius]
        x = [point.x for point in within_circle]
        y = [point.y for point in within_circle]
        plt.plot(x, y, 'r.', markersize=3)
        plt.show()


if __name__ == '__main__':
    qt = QTree([Point(random.normal(0, 2), random.normal(0, 2)) for x in range(200)], 10)
    qt.graph()
    print('Done')
