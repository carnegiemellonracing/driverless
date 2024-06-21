==============
File Structure
==============

This is how the controls subdirectory of is laid out.

.. list-table:: File Structure of ``controls/src``
   :widths: 25 25 25
   :header-rows: 1

   * - Folder
     - File Name
     - Description
   * - /
     - constants.hpp
     - Various CPU constants for controller, notably tuning parameters.
   * -
     - cuda_constants.cuh
     - A few constants for CUDA code.
   * -
     - types.hpp
     - Type definitions for CPU code, such as ROS messages.
   * - nodes
     - controller.hpp
     - Definition of ``ControllerNode`` class, main ROS node.
   * -
     - controller.cpp
     - Implementation of ``ControllerNode`` class. Main function that spins the node.
   * - mppi
     - mppi.hpp
     - Definition of abstract class ``MppiController``. ROS2 interface for MPPI.
   * -
     - mppi.cuh
     - Definition of ``MppiController_Impl`` class, where MPPI is implemented.
   * -
     - mppi.cu
     - Implementation of ``MppiController_Impl`` class.
   * -
     - functors.cuh
     - Helper functors (C++ function objects) to be used in Thrust higher-order functions in MPPI.
   * -
     - types.cuh
     - Definition of types used in MPPI.
   * - state
     - state_estimator.hpp
     - Definition of abstract class ``StateEstimator``. ROS2 interface for state estimation.
   * -
     - state_estimator.cuh
     - Definition of ``StateEstimator_Impl`` class, where state estimation is implemented.
   * -
     - state_estimator.cu
     - Implementation of ``StateEstimator_Impl`` class.
   * - cuda_globals
     - cuda_globals.cuh
     - Symbols that exist in GPU global memory, namely inertial state and curvilinear lookup table.
   * -
     - cuda_globals.cu
     - Initialization of read-only symbols, such as covariance matrix for MPPI's Brownian generation.
   * -
     - helpers.cuh
     - Helpers to work with the cuda_globals, specifically to utilize the lookup table.
   * - model
     - slipless/model.cuh
     - Slipless dynamics model. Currently used by mppi and state.
   * -
     - bicycle/model.cuh
     - Bicycle model that assumes wheel slip. Currently unused.
   * -
     - dummy/model.cuh
     - Very basic dummy model to test MPPI. Currently unused.
   * -
     - two_track/*
     - Complex and slow model. Currently unused.
   * - utils
     - cuda_utils.cuh
     - Utilities for CUDA/Thrust code
   * -
     - gl_utils.hpp
     - Declarations of utilities for OpenGL code.
   * -
     - gl_utils.cpp
     - Implementation of ``gl_utils.hpp``
   * - display
     - display.hpp
     - Declaration of functions for the debugging display (built with -D)
   * -
     - display.cpp
     - Implementation of ``display.hpp``.

==============
Namespaces
==============
Namespaces roughly correspond to the file structure

**nodes**: Code concerning the overall controller ROS node.

**mppi**: Code concerning MPPI, including helper functors.

**state**: Code concerning state estimation.

**cuda_globals**: Code concerning anything on GPU global memory. CUDA code only.

**utils**: Miscellaneous utilities.