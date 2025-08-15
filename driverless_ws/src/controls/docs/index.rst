.. driverless_controls documentation master file, created by
   sphinx-quickstart on Wed Jun 12 11:18:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. todo fill out links to other documentation

.. _Perceptions: https://cmr.red/perceptions-docs
.. _Path Planning: https://cmr.red/planning-docs
.. _C++: https://cplusplus.com/doc/tutorial/
.. _ROS2 Humble: https://docs.ros.org/en/humble/index.html
.. _CUDA: https://docs.nvidia.com/cuda/
.. _Thrust: https://nvidia.github.io/cccl/thrust/

============
Introduction
============

As part of our efforts to support other FSAE teams as they embark on their driverless journeys,
we have open-sourced our workspace at our `GitHub <https://github.com/carnegiemellonracing/driverless>`_ and documented them here.

Controls is one element of our software stack, coming in after `Perceptions`_ and `Path Planning`_.

We recommend heading to :doc:`Overview <source/explainers/overview>` for a top-down deep dive into how the controller works.
You can stop at any level of abstraction you prefer. While we only assume basic programming knowledge, feel free to read more into the software
that the controller depends on, namely `C++`_, `ROS2 Humble`_, `CUDA`_, and `Thrust`_.
We also provide a cheat sheet of common :doc:`Terminology </source/explainers/terminology>`.

If you're part of the team, looking to contribute or learn more about implementation details, check out :doc:`File Structure <source/explainers/file_structure>`
to understand how the codebase in ``driverless/driverless_ws/src/controls`` is laid out, then refer to the :doc:`API Reference <source/reference/api_reference>`

Have fun and enjoy the ride!

.. image:: images/car.*

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/explainers/overview
   source/reference/implementation
   source/explainerslist.rst
   source/explainers/terminology

