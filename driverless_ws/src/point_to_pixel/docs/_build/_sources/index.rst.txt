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

This documentation is meant to provide an introduction and high level conceptual overview to our perceptions pipeline. 
For more detail, including source code, please visit our `GitHub repository <https://github.com/carnegiemellonracing/driverless>`_ 

Checkout the links below to learn more about our sensor stack and our main algorithms!


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   source/overview/overview
   source/explainers/explainers
   source/implementation/api_reference
