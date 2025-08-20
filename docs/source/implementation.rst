Implementation
=====================

:term:`SLAM` Nodes
---------------------

There are three primary nodes for :term:`SLAM`. 

1. **real_data_slam_node_gps**

2. **real_data_slam_node_no_gps**

3. **controls_sim_slam_node**

The first two nodes are used to run on real data depending on whether we have :term:`GPS` or not. The last node is used to test our :term:`SLAM` implementation in-house with a :term:`rosbag` recorded on our in-house controls sim. 


Execution and Data flow
----------------------------

**Where does** :term:`iSAM2` **run (CPU/GPU)?**

The :term:`iSAM2` :term:`SLAM` implementation runs entirely on the CPU. It is written in C++ and uses :term:`GTSAM`, a CPU-optimized factor graph library. No GPU acceleration is used, as :term:`iSAM2` is designed for incremental updates that are efficient enough for real time execution on modern multi-core CPUs. Our :term:`SLAM` nodes run within a :term:`ROS 2 Node` written in C++ and leverage threading when available (through :term:`TBB`), although much of the computation remains serial due to the incremental nature of the updates.

**What is the data flow?**

.. image:: ./img/slam_data_flow.png
    :align: center

1. Input: 

- Cone observations from the perceptions pipeline (2D positions of blue/yellow cones)

- :term:`IMU` and velocity estimates

- :term:`Pose` estimates from :term:`GPS` (optional)

2. Processing: 

- Each cone observation is added as a :term:`bearing-range factor` connecting the vehicle :term:`pose` to cone :term:`landmark`\ s

- The current vehicle pose (if available) is added as a new variable in the :term:`Factor Graph`

- Motion priors are added between sequential poses (using :term:`IMU`/velocity)

- This information is fed into :term:`iSAM2` which incrementally updates the estimates of all previous poses and :term:`landmark` (cone) positions

3. Output: 

- Estimated vehicle :term:`pose` (2D position and orientation) at each time step

- Updated cone map as global positions of all :term:`landmark`\ s (cones)

Additional Considerations
--------------------------
While the current :term:`SLAM` and path planning system is designed to be efficient and robust, several challenges may arise in practice:

    Sensor Noise and Drift: :term:`GPS` and :term:`IMU` measurements can be noisy or suffer from drift. This can lead to inaccurate localization or erroneous map updates.

    :term:`Data Association` Ambiguity: Identifying whether a newly observed cone corresponds to a previously seen cone (:term:`data association`) is challenging in cluttered environments. Misassociations can degrade map quality and cause localization accuracy.

    Tuning Sensitivity: :term:`SLAM` performance heavily depends on parameters such as the :term:`Mahalanobis distance` threshold and noise models. Poorly tuned parameters can result in either missed associations or false positives.

    Failure Recovery: If the system becomes mislocalized or the :term:`factor graph` diverges due to accumulated error, recovering gracefully is non-trivial and may require additional loop closure strategies or reinitialization.

    Real World Edge Cases: Lighting conditions, partial occlusions, or uneven terrain can cause discrepancies between perception and ground truth, which affect downstream :term:`SLAM` and planning modules.