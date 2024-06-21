=======================
MPPI Algorithm Overview
=======================

The MPPI (Model Predictive Path Integral) algorithm is a model predictive control algorithm that uses a stochastic optimization approach to find the
optimal control action for a given task.

We highly recommend reading the :doc:`State Estimation Overview <state_estimation>` before proceeding.

Motivation
----------
Because our dynamics model is not differentiable and the task is non-convex, we cannot use gradient-based optimization
methods. We chose MPPI on the recommendation of our advisor due to its speed and simplicity. Furthermore, MPPI is
highly parallelizable since each sample can be computed independently, allowing us to exploit our onboard GPU.

.. todo Which paper did we take inspiration from?

Technical Primer
----------------
We use the following terms to describe the MPPI algorithm:

* State: The state of the vehicle. We define this to be x, y, yaw and speed.
* Control Action: The control action to be executed by the vehicle. We define this to be steering wheel angle and wheel throttle.
* Action Trajectory: An array of control actions, representing a temporal sequence of control actions to be executed every ``controller_period``.
* Dynamics Model: A model that future state after a given time, given the current state and control action.
* Cost Function: A function that evaluates the quality of a given state. The goal is to minimize this function.
* Cost-to-go: The total cost of being in a given state and all the states that follow it.
* Controller Period: The time interval between control actions.
* Perturbation: Brownian noise centered at 0 that we add to a base action trajectory to generate a sample.

Algorithm
---------
Execute these in a continuous loop:

1. Obtain the inertial state of the vehicle.
2. Perturb the previous "best" action trajectory with Brownian noise repeatedly to generate ``num_samples`` action trajectories.
3. For every sample, do state rollouts. Repeatedly apply the dynamics model to the state and perturbed control action to generate a sequence of states.
4. For every sample, evaluate the cost-to-go for every state in the sequence.
5. For every time step, compute the weighted average of the perturbed control actions over all the samples, based on the associated cost-to-go and the likelihood of the perturbation (see :ref:`importance_sampling`). This is the optimal action trajectory.

6. Execute the first control action in the optimal action trajectory, then store the rest of the trajectory for the next iteration of MPPI.

For a more technical specification, see `here <../../_static/mppi.pdf>`_.

Implementation
--------------
The controller node owns an outward facing :doc:`MppiController <../reference/class/MppiController>`, which has one main member function ``generate_action()``
that outputs a control action. This is defined in ``controls/src/mppi/mppi.hpp``.

This however, is only an abstract base class. The actual implementation is in the derived class, :doc:`MppiController_Impl <../reference/class/MppiController_Impl>`, which is
defined in ``controls/src/mppi/mppi.cuh``.

.. note:: Why 2 classes?

    MPPIController_Impl needs to call Thrust functions, so the class must be defined in a .cuh file with code in the corresponding .cu file. However, ROS is CPU-bound, so it
    can only interface with CPU-only classes in .hpp files. Thus, MPPIController is an abstract base class that
    provides member functions for the controller node to call, but in reality is an instance of MPPIController_Impl.

MPPI makes heavy usage of Thrust to conduct operations in parallel over the GPU. Copying data from the CPU to the GPU
or vice versa is an expensive operation, so we avoid it as much as possible. The MPPIController_Impl reads in the estimated
current inertial state and the inertial-to-curvilinear lookup table as calculated by the StateEstimator (and already exist on the GPU). It creates and
manipulates device vectors such as ``m_action_trajectories`` on the GPU,
then copies over a single control action to the CPU to pass in a message.

Thrust relies on small unary or binary functions to do :ref:`maps or reductions <hofs>`. Since we need to capture pointers to
our device vectors, we wrap our functions in C++ functors. These functors, along with helper functions to be run on the
GPU, such as the dynamics model, are defined in ``controls/src/mppi/functors.cuh``. The main loop of
MPPI is implemented in ``generate_action()`` of ``controls/src/mppi/mppi.cu``, as shown here:

.. image:: /images/mppi.*
    :width: 100%
    :align: center

.. _importance_sampling:

Importance Sampling
-------------------

The MPPI algorithm uses importance sampling to calculate weights for each control action.

The weight for a control action :math:`u` is given by :math:`e^{-\frac{1}{\lambda}J - D}` where :math:`J` is the associated
cost-to-go, and :math:`D` is the natural log of the probability density of the associated sampled perturbation, where the
probability distribution is the multivariate normal distribution. The parameter :math:`\lambda` is a temperature parameter
that can be tuned to balance model convergence and noise.

Multiplying by the exponent of the negative log is equivalent to dividing by the probability density. More unlikely samples
are given higher weight, since they explore the search space more.

.. _hofs:

Higher Order Functions
----------------------

Work refers to the time taken to execute all the operations of a given task by a single processor.
Span refers to the minimum time taken to execute the task's operations in parallel across infinite processors.

.. <insert mapping image>

Given a collection of elements, a map applies a unary mapping function to each element to produce a new collection of elements.
Assuming the mapping function is O(1), the map operation has work O(n) but span O(1).

.. <insert reduce image>

Given a collection of elements, a reduce applies a binary reduction function to combine all the elements into a single
accumulated value. Assuming the reduction function is O(1), the reduce operation has work O(n) but span O(log n).

Alterations
-----------
Consider the following alterations to our implementation:

* State can capture more information about the vehicle
* Control action can be more expressive to allow the algorithm to learn behaviors such as torque vectoring. Granted, this requires a more complex dynamics model.
* The dynamics model can be learned from data.
* The cost can be made to be a function of both state and control action.

