# This package is a refactored version of the pipeline used on 24a

- ** find the original code inside the `HesaiLidar_ROS_2.0/src` LiDAR driver (if it's still in our repo) **

## Algorithms

1. Ground filtering: Grace and Conrad 

2. Clustering: DBSCAN

3. Removal of extraneous clusters: DBSCAN


## TODO

- create structure that supports using and turning off kd trees via flag
    - look into [nanoflann](https://jlblancoc.github.io/nanoflann/)

- implement algorithms

- test