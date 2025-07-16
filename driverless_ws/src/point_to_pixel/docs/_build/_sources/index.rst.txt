.. cmr_perceptions_24a_point_to_pixel documentation master file, created by
   sphinx-quickstart on Tue Jul 15 22:05:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _Path Planning: https://cmr.red/planning-docs
.. _Controls: https://cmr.red/controls-docs

========================================
CMR Driverless Perceptions Stack for 24a
========================================

.. NOTE::
   PLACEHOLDER FOR IMAGE of car w/ sensors

What is Perceptions?
========================================

Perceptions constitutes the system of the car that ingests and interprets sensor data from the environment. 
Via this interpretation, the perceptions module enables `Path Planning`_ and `Controls`_ modules to make high-level
decisions to control the car.

We employ a variety of sensors to accomplish this task:

- `HESAI AT128 Solid State LiDAR <https://www.hesaitech.com/product/at128/>`_
- `Dual ZED2 Stereo Cameras (only used for RGB frames) <https://www.stereolabs.com/products/zed-2>`_
- `MTi-680G RTK GNSS/INS GPS <https://www.movella.com/products/sensor-modules/xsens-mti-680g-rtk-gnss-ins>`_

Using these three sensors we efficiently generate an accurate local view of the track and cones. 

This documentation is meant to provide an introduction and high level conceptual overview to our perceptions pipeline. 
For more detail, including source code, please visit our `GitHub repository <https://github.com/carnegiemellonracing/driverless>`_ 


What data do we work with and where does it go?
========================================

Our LiDAR, a HESAI AT128 hybrid solid-state sensor is our primary source of depth information. Through the LiDAR we ingest a 
`point cloud <https://en.wikipedia.org/wiki/Point_cloud>`_. We employ several processing algorithms (see `explainers <source/explainers/cpp_processing>`_)
eventually resulting in a set of points that represent the centroid of cones on the track in front of us.

.. NOTE::
   PLACEHOLDER FOR IMAGE of LiDAR?

From our cameras, we get our primary source color information. Through a `direct linear transform <https://en.wikipedia.org/wiki/Direct_linear_transformation>`_ \
we color our cone centroids from the previous step and pass them down the pipeline to `Path Planning`_ and `Controls`_.

.. NOTE::
   PLACEHOLDER FOR IMAGE of LiDAR?


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   source/api_reference




