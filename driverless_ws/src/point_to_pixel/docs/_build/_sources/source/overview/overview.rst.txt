.. _Path Planning: https://cmr.red/planning-docs
.. _Controls: https://cmr.red/controls-doc

=============
Overview
=============

The Perceptions Problem
=======================
Given a track delimited by yellow and blue cones, we must reliably and efficiently determine the track and cones and pass down the driverless pipeline.

.. figure:: trackdrive_diagram.png
    :width: 600
    :align: center

    *Figure taken from 2025 Formula Student Germany Competition Handbook.*
    

Sensors
=======

We employ a variety of sensors to accomplish this task:

- `HESAI AT128 Solid State LiDAR <https://www.hesaitech.com/product/at128/>`_
- `Dual ZED2 Stereo Cameras (only used for RGB frames) <https://www.stereolabs.com/products/zed-2>`_
- `MTi-680G RTK GNSS/INS GPS <https://www.movella.com/products/sensor-modules/xsens-mti-680g-rtk-gnss-ins>`_

Using these three sensors we efficiently generate a local view of the track and cones. 

What data do we work with and where does it go?
===============================================

LiDAR Module
------------

Our LiDAR, a HESAI AT128 hybrid solid-state sensor is our primary source of depth information. Through the LiDAR we ingest a 
`point cloud <https://en.wikipedia.org/wiki/Point_cloud>`_. We employ several processing algorithms (see :doc:`explainers <source/explainers/lidar_module>`)
eventually resulting in a set of points that represent the centroid of cones on the track in front of us.

.. NOTE::
   | PLACEHOLDER FOR IMAGE OF LiDAR?
   | DOUBLE CHECK THE OUTPUT TYPES / TOPIC NAMES


To avoid overhead from publishing the entire LiDAR point cloud to a ROS topic, we integrated our lidar_module code into the ROS driver available with our LiDAR.

    Output: a set of cone centroids. It is a message type from our custom ROS2 ``interfaces`` package

    * ``/perc_cones``

        * ``interfaces::msg::ConeList``   

    
Coloring Module
---------------

From our RGB cameras we get our primary source color information. Though we used two stereolabs ZED cameras which have stereoscopic capability,
we opted to avoid any depth processing due to latency concerns. Instead we just use the cameras for rgb images.

Through a `direct linear transform <https://en.wikipedia.org/wiki/Direct_linear_transformation>`_
we color our cone centroids from the previous step and pass them down the pipeline to `Path Planning`_ and `Controls`_.

.. NOTE::
   | PLACEHOLDER FOR IMAGE OF CAMERAS?
   | DOUBLE CHECK THE OUTPUT TYPES / TOPIC NAMES

Our coloring module is housed in our custom ROS2 package, :doc:`point_to_pixel <../implementation/coloring_module>`. 

    Output: a set of colored cone centroids

    * ``/colored_cones``

        * ``interfaces::msg::PPMConeArray``