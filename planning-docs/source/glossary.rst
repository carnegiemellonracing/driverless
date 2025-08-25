Glossary
===========

.. glossary::

   SLAM
     Simultaneous Localization and Mapping. A method used by autonomous systems to build a map of an unknown environment while simultaneously keeping track of the system's own position within that environment.

   GTSAM
     Georgia Tech Smoothing and Mapping (GTSAM) is a C++ library, with Python bindings, that implements smoothing and mapping using factor graphs. It is widely used in robotics and computer vision for tasks such as SLAM (Simultaneous Localization and Mapping) and Structure from Motion (SfM), enabling efficient estimation of trajectories and landmark positions through probabilistic inference.

   iSAM2
     A specific algorithm for SLAM (Incremental Smoothing and Mapping, version 2). It uses a factor graph representation and performs efficient incremental updates to optimize the vehicle's pose and the map of the environment.

   SVM
     Support Vector Machine. A supervised machine learning algorithm used for classification and regression tasks. It finds the optimal hyperplane that separates data into distinct classes with maximum margin.

   Factor Graph
     A bipartite graph consisting of variable nodes and factor nodes.

     - **Variable nodes** represent unknowns, such as the vehicle pose or landmark positions.
     - **Factor nodes** represent probabilistic constraints between variables, such as measurements or priors.

     The factor graph encodes the full probabilistic model used for SLAM.

   Joint probability distribution
     A statistical measure that gives the probability of two or more random variables occurring simultaneously. In SLAM, it represents the combined likelihood of vehicle poses and landmark positions given all observations and constraints.

   Joint probabilistic distribution
     See :term:`joint probability distribution`.

   Variable Node
     A node in the factor graph representing a variable to be estimated, such as a vehicle pose (`x_n`) or landmark position (`l_n`).

   Factor Node
     A node that connects one or more variable nodes and encodes a probabilistic constraint, such as a sensor measurement.

   Prior Factor
     A special type of factor node used to initialize the estimation problem with known values (e.g., the first car pose).

   Bearing-range factor
     A type of measurement used in SLAM that combines the *bearing* (angle) and *range* (distance) between the robot and a landmark. This factor relates the robot's pose to the landmark's position and is commonly used in factor graph optimization to improve landmark and pose estimation accuracy.

   Mahalanobis Distance
     A distance metric that accounts for uncertainty in measurements. It measures the number of standard deviations a point is from the mean of a distribution. Used for data association in SLAM.

   Data Association
     The process of matching observed landmarks (e.g., cones) with previously seen landmarks in the map. Essential for consistent mapping and localization.

   IMU
     An Inertial Measurement Unit (IMU) is a sensor device that measures and reports a vehicle's specific force, angular rate, and sometimes magnetic field. It typically contains accelerometers, gyroscopes, and sometimes magnetometers, and is used to estimate orientation, velocity, and motion of the system it is attached to.

   GPS
     Global Positioning System (GPS) is a satellite-based navigation system that provides location and time information anywhere on or near the Earth. It is commonly used in robotics to obtain position estimates for localization and navigation.

   Pose
     The position and orientation of the vehicle in the global frame. Often represented as (x, y, θ) in 2D SLAM.

   Quaternion
     A 4D representation of 3D orientation that avoids singularities and discontinuities. Consists of four components: (w, x, y, z).

   Twist
     A message (in ROS) containing the linear and angular velocity of a vehicle, used for estimating motion.

   Odometry
     The use of motion sensors (such as encoders, IMU, or GPS) to estimate the position and orientation of a robot over time.

   Landmark
     A static feature in the environment used for localization, such as a cone in Formula Student Driverless.

   ROS (Robot Operating System)
     An open-source framework for building robotic systems. ROS provides tools, libraries, and conventions for writing modular robot software, including message-passing between processes (nodes), hardware abstraction, and device drivers. In this project, ROS 2 is used to implement and run various nodes such as SLAM, perception, and control.

   ROS 2 Node
     A process that performs computation in the ROS 2 framework. In this context, SLAM nodes are responsible for managing factor graphs, receiving sensor data, and publishing localization estimates.

   Rosbag
     A file format and tool in ROS used to record and replay messages from topics. Rosbags  allow for offline analysis, debugging, or simulation.

   TBB 
     (Threading Building Blocks) A C++ library used for parallel programming. It can improve performance in multi-threaded SLAM systems.