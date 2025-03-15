====================
Controller Overview
====================

.. image /images/controller_implementation.*

If you're unfamiliar with ROS2, check out https://docs.ros.org/en/humble/Tutorials.html.

The controller hinges on 3 things:

* A novel :doc:`state estimation <state_estimation>` pipeline.
* Our state of the art implementation of the :doc:`MPPI algorithm <mppi_algorithm>`.
* An efficient :doc:`dynamics model </source/explainers/slipless_model>` that is used for both state estimation and MPPI.

This is implemented in two objects that the node **owns** - the :doc:`StateEstimator <../reference/class/StateEstimator>` and the :doc:`MppiController <../reference/class/MppiController>`.

They provide the following functionality:

* StateEstimator: given twist and spline, estimates inertial state and a inertial to curvilinear lookup table.
* MPPIController: given inertial state and the lookup table, calculates the optimal control action to take.

The inertial state and curvilinear lookup table are shared through GPU global memory.

This diagram shows this relationship:

.. image:: /images/controller_interface.*

Implementation
--------------

As for the node itself, these are the implementation details:

**Initialization**: Construction creates subscribers and publishers, then launches MPPI in a new thread.

**Callbacks**

* Spline: Updates the state estimator with the most recent set of spline points. Notifies MPPI thread that the state is dirty.
* Twist: Updates the state estimator with the most recent twist. Notifies MPPI thread that the state is dirty.

These callbacks are mutually exclusive.

**MPPI Thread**: Loops continuously. Waits to be notified that state is dirty (i.e. new incoming message).
When notified, does state estimation, then runs MPPI. Publishes the control action and controller info.

Class reference can be found in :doc:`../reference/class/ControllerNode`.
Code for the node can be found in ``controls/src/nodes/controller.cpp``.

.. linear velocity from twist, yaw rate from steering wheel angle, better than time-syncing
  baked into model


.. The action is double buffered, to minimize the delay that MPPI will have on action publishing. The
    timer callback is parallel with any other callbacks, so while the consistency of the publishing isn't
    guaranteed, it won't be delayed by MPPI or state updates.
    add link to double buffering, inquire about consistency of publishing

