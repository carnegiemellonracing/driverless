Overview
=========

What is Path Planning?
-----------------------------------
Path planning is responsible for determining the route the car should follow to navigate the track. It has two main responsibilities:

1. **Midline Calculation**

This refers to the process of generating a route between the blue and yellow cones observed at the current time step. The midline is necessary when the car is competing its first lap around the track because without knowledge of the track's layout, the car is forced to naively navigate the track. 

2. :term:`SLAM` **(Simultaneous Localization and Mapping)**

As the car completes its first lap, a :term:`SLAM` algorithm is employed to refine its position in the track and the position of previously seen cones (localization) based on newly observed cones while also mapping out the racetrack (mapping).

What data does it receive?
---------------------------
Path planning receives cone detections (positions of blue and yellow cones) from the perceptions pipeline. These cone positions are observed using LiDAR and processed in real-time. It also receives velocity, orientation, and position in the form of :term:`twist`, :term:`quaternion`, and :term:`pose`. These readings are filtered and come from the :term:`IMU` and :term:`GPS`. 

.. note:: 
   - :term:`twist` gives us linear and angular velocity
   - :term:`quaternion` tells us yaw, pitch, and roll
   - :term:`pose` gives us position and orientation

   These components combined allow us to reason about the motion of the car as well as the relative position of cones. 

What does it output, and to what system?
-----------------------------------------
The output is a set of path waypoints (from midline calculation) and a :term:`SLAM`-generated map. The path is sent to the controls pipeline, which uses it to generate control actions for the car. The map is used in later laps to improve navigation. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   implementation
   explainers
   isam2
   glossary
