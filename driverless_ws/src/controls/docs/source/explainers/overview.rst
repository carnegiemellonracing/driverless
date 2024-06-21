============
Overview
============

What is Controls?
-----------------

Controls is the process of taking in information about the vehicle's **state**, e.g. its position and velocity, in relation
to the outside **world**, e.g. the track set out by the cones, and calculating the optimal **control action** to take,
e.g. throttle and steering wheel angle.

How does it work?
-----------------

Controls is our custom ROS2 package that contains the controller |Node|.

It subscribes to the following topics:

* ``spline``: An array of points in space calculated from |Path Planning|.
* ``filter/twist``: 6-dimensional :doc:`twist </source/explainers/terminology>` information from the IMU built into our `Movella MTi-680G RTK GNSS/INS GPS <https://www.movella.com/products/sensor-modules/xsens-mti-680g-rtk-gnss-ins>`_.

It publishes to the following topics:

* ``control_action``: The calculated optimal control action to be sent to the |Actuators| node.
* ``controller_info``: Information about the controller, for debugging purposes.

Setup
-----------------
To build the controls package, run the following command from inside the ``driverless/driverless_ws`` directory:

.. code-block:: bash

    ./build-controls.py

To run the controller node, run:

.. code-block:: bash

    ros2 run controls controller

Next Up
-------

To learn more about the controller, visit :doc:`here </source/explainers/controller>` for a detailed overview.
