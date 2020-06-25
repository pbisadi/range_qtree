# range_qtree
This is a python version of the QuadTree that supports efficient aggregated ranges queries on geometric data points.

Ex. To answer what is the average salary in 10km around a house (coordination), first you need to find all data points within the range and then apply the average function on their salary attribute. If this query is run once, all you need to do it effiently is a data structure like QTree or KDTree to fetch the points. But what if the query is: The average salary of 10km radius neighbourhood of each house? This code is for improving the performance of such query by preprocessing the data. A query that contains an aggregate function about data points within a range.
