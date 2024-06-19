Class state::StateEstimator
=====================================

In a nutshell, given twist and spline information through ROS messages, it estimates:

- the current inertial state of the vehicle (by using our `model <../../../_static/model.pdf>`_ to project from stale data)
- a lookup table from inertial state to curvilinear state based on the spline and OpenGL magic

Then it syncs these to the GPU for the MPPI controller to use.

This is merely an abstract base class that provides functions to the controller node. The actual implementation is
within the derived class, :doc:`StateEstimator_Impl`. Implementation details are documented there.

.. note:: Why 2 classes?

    StateEstimator_Impl needs to launch CUDA kernels to do its estimation, so the class must be defined in a .cuh file
    (*controls/src/state/state_estimator.cuh*) with code in the corresponding .cu file. However, ROS is CPU-bound, so it
    can only interface with CPU-only classes in .hpp files. Thus, StateEstimator is an abstract base class that
    provides member functions for the controller node to call, but in reality is an instance of StateEstimator_Impl.

.. doxygenclass:: controls::state::StateEstimator

.. toctree::
    StateEstimator_Impl
    StateProjector