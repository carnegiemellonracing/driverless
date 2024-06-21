==============
File Structure
==============

An explanation of the file structure of the controls repository.

.. list-table:: File Structure of ``controls/src``
   :widths: 25 25 50
   :header-rows: 1

   * - Folder
     - File Name
     - Description
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
   * -

.. list-table:: File Structure of controls/
   :widths: 25 75
   :header-rows: 1

   * - File Name
     - Description
   * - `file1`
     - Description of `file1`
   * - `file2`
     - Description of `file2`
   * - `file3`
     - Description of `file3`
   * - `file4`
     - Description of `file4`