LiDAR Module Concepts
=====================

Overview
--------

A LiDAR sensor, while providing fast accurate depth information, presents some 
unique challenges for our perceptions stack. We must first filter out a good amount of extraneous information
in order to get to the main perceptions goal: locations of cones in 3d space.

The LiDAR Module employs two main algorithms to take in unstructured point clouds from the environment,
efficiently filter out the ground and sky, and finally identify clusters of points representing the centroids of cones.

.. Note::
    Add Module Diagram with pictures from each stage. (Use my python package for pipeline sim)


:doc:`Grace and Conrad <../implementation/lidar_module>`
--------------------------------------------------------

.. Note::
    Add the following diagrams:

    - Image of points before and after GNC
    - depiction of binned radius azimuth coordinates
    - car frame
    - GNC algorithm diagram

    Figure out where GNC name actually came from...

We use a ground filtering algorithm called Grace and Conrad (developed by grace and conrad from our team a few years back).
Essentially, we split our point cloud into bins, find the point with the minimum radius and z in and fit a plane to a 
RANSAC-based sample of those points. We conclude by filterting out points above and below that plane by a tuneable height parameter.


:doc:`Density-Based Spatial Clustering of Applications with Noise (DBSCAN) <../implementation/lidar_module>`
------------------------------------------------------------------------------------------------------------

.. Note::
    Add the following diagrams:

    - Imaages of points before and after DBSCAN
    - DBSCAN clusters
    - DBSCAN algorithm diagram

After filtering out the ground, we make the assumption that all clusters of sufficient density left represent cones. This may be 
a bold assumption to make, but we find when combined with another run of DBSCAN (we call it DBSCAN2 in our codebase) most extraneous 
objects are removed. DBSCAN is implemented by iterating through each point in the cloud and identifying its neighborhood within a specified radius (ε). 
If a point has at least a minimum number of neighbors (MinPts), it becomes a core point and forms a cluster by recursively including its density-reachable neighbors. 
Points that do not meet this criterion and are not reachable from any core point are filtered out.

- `DBSCAN Wikipedia reference <https://en.wikipedia.org/wiki/DBSCAN>`_
- `DBSCAN original paper <https://dl.acm.org/doi/10.5555/3001460.3001507>`_