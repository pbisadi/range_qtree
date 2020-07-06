import string

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pandas as pd
from matplotlib import patches
import functools as ft


class Node:
    id_gen_counter = 0

    def __init__(self, df, dimension_cols, center=None,
                 widths=None, threshold=100, parent_node=None):
        """
        Store the row indexes in df balanced based on the center point and coordinate_cols of those rows.
        Each node has a N-dimensional box assigned to it which has upper and lower bound with equal distance
        to the center point (Width).

        Args:
            df(DataFrame): The DataFrame containing the coordinate columns and the data.
            dimension_cols(list): The list of coordinate columns in df parameter.
            center(list): The center point of covered box by this node.
                Will be calculated based on the minimum and maximum values of each dimension column it it is None.
            widths(list): The distance between the center point and upper and lower bound.
                Will be calculated based on the minimum and maximum values of each dimension column it it is None.
            threshold: Maximum number of points that each leaf node can store
        """

        # For debugging
        Node.id_gen_counter += 1
        self._id = Node.id_gen_counter
        self.parent = parent_node

        minimums = df[dimension_cols].min()
        maximums = df[dimension_cols].max()

        self.center = center if center else [(l + u) / 2 for (l, u) in zip(minimums, maximums)]
        """The coordinates of the center point"""

        self.widths = widths if widths else [(u - l) / 2 for (l, u) in zip(minimums, maximums)]
        """The distances from center"""

        self.threshold = threshold
        """Maximum number of points that each leaf node can store"""

        self.df = df
        """The DataFrame storing the data including the coordinates columns"""

        self.coordinate_cols = dimension_cols
        """List of columns defining the dimensions for each data point coordination in order"""

        self.sub_nodes = None
        """None if it is a leaf node. Otherwise contains 2 ^ D children"""

        if len(df) > threshold:
            self.assign_node_index()
            self.sub_nodes = [None] * (2 ** len(self.center))

            # TODO: Use decimal for more accuracy
            sub_nodes_width = [w / 2 for w in self.widths]
            for i, _ in enumerate(self.sub_nodes):
                self.sub_nodes[i] = Node(
                    df=self.df.loc[self.df["__sub_node_idx"] == i],
                    dimension_cols=dimension_cols,
                    center=self._sub_node_center(i),
                    widths=sub_nodes_width,
                    threshold=threshold,
                    parent_node=self)

    def _sub_node_center(self, idx):
        """
        Adds or subtracts half of the width of this node from its center based on the idx
        to generate the center of the sub node
        """
        sub_node_center = self.center.copy()
        d = len(self.coordinate_cols)  # number of dimensions
        for i in range(d):
            if (1 << (d - i - 1)) & idx:
                sub_node_center[i] += self.widths[i] / 2
            else:
                sub_node_center[i] -= self.widths[i] / 2
        return sub_node_center

    def sub_node_idx(self, data_point_idx: int) -> int:
        """
        Get the index of the sub node which provided data_point belongs to.
        This function is the core logic of QTree data structure

        Args:
            data_point_idx: The index of the data point in self.df

        Returns:
             int: The index of the chile that contains (or must) the provided data_point_idx

        """
        # TODO: Think of a way to calculate the child indexes as a batch instead of one by one.
        #       For example address coordination columns in df

        sub_node_index = 0
        for i in range(len(self.coordinate_cols)):
            sub_node_index = (sub_node_index << 1) | int(
                self.center[i] <= self.df[self.coordinate_cols[i]].iloc[data_point_idx])
        return sub_node_index

    def all_leaf_nodes(self, initializer=None):
        """
        get all descendant leaf nodes

        Args:
            initializer(list): the result will be added to the initializer list of provided

        Returns:
            list: all descendant leaf nodes of the current node
        """
        if initializer is None:
            initializer = list()

        if not self.sub_nodes:
            initializer.append(self)
        else:
            for n in self.sub_nodes:
                n.all_leaf_nodes(initializer)

        return initializer

    def assign_node_index(self):
        """Assigns each coordinate to one of the sub nodes"""
        self.df = self.df.assign(__sub_node_idx=0)

        bit_significance = 0
        for i, c in enumerate(self.coordinate_cols):
            bit_significance += 1
            self.df["__sub_node_idx"] += (self.center[i] <= self.df[c]) * 2 ** bit_significance


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

        self.root = Node(df, coordinate_cols, threshold=self.threshold)

    def aggregate_nodes(self):
        pass

    def range_query(self, center, radius):
        pass

    def graph(self):
        fig = plt.figure(figsize=(12, 8))
        plt.title("Quad-Tree")
        ax = fig.add_subplot(111)
        leaves = self.root.all_leaf_nodes()
        print(f'Number of segments: {len(leaves)}')
        areas = set()
        for n in leaves:
            areas.add(ft.reduce(lambda s, w: s * (w * 2), n.widths, 1))
        print(f'Minimum   segment area: {min(areas)} units')

        # plot all points
        for n in leaves:
            ax.add_patch(
                patches.Rectangle((n.x0, n.y0), n.widths[0], n.height, fill=False, color='gray', linewidth=0.2))
        plt.plot(self.df['x'], self.df['y'], linestyle='', color='gray', marker='.', markersize=1)

        # plot query
        query_point = np.array([2, 2])
        query_radius = 2
        ax.add_artist(plt.Circle(tuple(query_point), query_radius, color='#FF000033'))
        within_circle = self.df.loc[
            self.df[np.linalg.norm(np.array([self.df['x'], self.df['y']]) - query_point) < query_radius]]
        plt.plot(within_circle['x'], within_circle['y'], 'r.', markersize=3)
        plt.show()


if __name__ == '__main__':
    row_count = 2000

    df = pd.DataFrame({
        'x': [random.normal(0, 2) for _ in range(row_count)],
        'y': [random.normal(0, 2) for _ in range(row_count)],
        'name': [string.ascii_uppercase[random.randint(0, 26)] for _ in range(row_count)]
    })

    qt = QTree(df, ['x', 'y'])
    qt.graph()
    print('Done')
