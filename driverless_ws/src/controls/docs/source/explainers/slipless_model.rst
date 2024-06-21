==============
Model Overview
==============

Our dynamics model takes in state and control input and outputs
the next state after a given timestep. We make a Markov assumption that the future state depends only on the current state and control input.

Notation
--------
* :math:`\vec{x}`: state vector
* :math:`\vec{u}`: control action vector
* :math:`\Delta t`: model timestep
* :math:`\vec{x_{next}}`: state vector after timestep
* :math:`\vec{f}`: dynamics function

* :math:`x`: x position in inertial frame
* :math:`y`: y position in inertial frame
* :math:`\psi`: yaw angle from x-axis
* :math:`v`: scalar speed of the car

* :math:`\delta`: steering angle
* :math:`\tau_{motor}`: total motor torque

* :math:`m`: mass of the car
* :math:`R`: wheel radius
* :math:`\hat{\delta}`: kinematic steering angle (explained later)
* :math:`\kappa`: understeer gradient
* :math:`\alpha`: slip angle (angle between velocity vector and car body)
* :math:`\omega`: angular velocity of the car

* :math:`r`: turning radius
* :math:`r_{rear}`: distance from rear wheel to center of turning circle
* :math:`r_{front}`: distance from front wheel to center of turning circle
* :math:`L_{front}`: distance between front wheel and center of mass
* :math:`L_{rear}`: distance between rear wheel and center of mass

* :math:`\tau_{front}`: torque on front wheel
* :math:`\tau_{rear}`: torque on rear wheel
* :math:`F_{front}`: force on front wheel
* :math:`F_{rear}`: force on rear wheel
* :math:`F_{drag}`: drag force (rolling and air resistance)
* :math:`F_{net}`: net force on car

Specification
-------------

.. math::

    \text{Given $\vec{x}$ } &= \begin{bmatrix}
        x \\
        y \\
        \psi \\
        v \\
    \end{bmatrix} \text{ and } \vec{u} = \begin{bmatrix}
        \delta \\
        \tau_{motor} \\
    \end{bmatrix}

    \text{Define the dynamics function $\vec{f}$ such that: }

    \vec{x_{next}} = \vec{f}(\vec{x}, \vec{u}, \Delta t)

    \text{where } \vec{x_{next}} = \begin{bmatrix}
        x_{next} \\
        y_{next} \\
        \psi_{next} \\
        v_{next} \\
    \end{bmatrix}


.. note::

    The :math:`x` axis points forward and the :math:`y` axis points to the right. The car starts facing the positive x direction.
    A positive yaw angle means the car is turning clockwise.

We use a modified bicycle model that introduces a couple of assumptions:

.. Note: this is discretized in time

.. rst-class:: numbered-list

1. There is no lateral slip on the tires.
2. Between model timesteps, the car moves in approximately uniform circular motion.
3. There is no longitudinal slip on the tires.
4. The angular inertia of the wheels is negligible compared to the inertia of the car and the torque being applied.

This assumption holds true unless the car is cornering **very** fast.

We will refer to the assumptions as [1], [2], [3] and [4].

During cornering, the car is modeled as such:

.. image:: /images/model.*
    :align: center

.. note::

    The car is not necessarily aligned to the x and y axes.

From [1], both tires move parallel to themselves with no sideway slip. Thus, given solely by the steering angle,
the turning radius and center should be calculable.

However, as speed increases, the car understeers more. The steering wheel needs to be turned more to achieve the same
turning radius. This is a consequence of our slipless assumption.

We characterize this with a tunable parameter called the understeer gradient :math:`\kappa`, and define the
kinematic steering angle :math:`\hat{\delta} = \frac{\delta}{1 + \kappa v}`, which is used to determine turning
radius and center.

.. note::

    The understeer gradient can be empirically measured by doing skidpad at various speeds and comparing the
    steering angle to the turning radius.

Calculate the slip angle :math:`\alpha`, which is the angle between the car's velocity vector and the car's body.

.. math::

    r_{rear} = (L_{front} + L_{rear}) / tan(\hat{\delta})

    \alpha = tan^{-1}(\frac{L_{rear}}{r_{rear}})

We calculate the higher order terms of :math:`\vec{x_{next}}` first - speed.

From [2], the car's velocity is perpendicular to the turning axis. Thus, to find :math:`\delta v`, we need to find the
net force on the car in the direction of :math:`\vec{v}`.

.. math::
    :class: center

    \begin{gathered}
    \tau_{front} \text{ and } \tau_{rear} \text{ are derived from } \tau_{motor}, \text{ gear ratio and drive mode.}

    \text{Because of [3] and [4], all torque goes into force on the wheels}

    F_{front} = \frac{\tau_{front}}{R} \text{ and } F_{rear} = \frac{\tau_{rear}}{R}

    F_{net} = F_{front}cos(\delta - \alpha) + F_{rear}cos(\alpha) - F_{drag}

    \delta v = \frac{F_{net}}{m} \Delta t

    v_{next} = | \vec{v} + \delta v |, \text{ since the car can't go backwards (negative torque is regenerative braking)}
    \end{gathered}

Actual steering angle is used here since that determines the direction of the forces.

Over the model timestep, the average speed :math:`\bar{v} = \frac{v + v_{next}}{2}`. Use this to recalculate :math:`\hat{\delta}, r, \alpha`.

Angular speed :math:`\omega = \frac{\bar{v}}{r}`, and because of [2], :math:`\frac{d\psi}{dt} = \omega`.

Thus, the new yaw angle :math:`\psi_{next} = \psi + \omega \Delta t`.

Finally, calculate the new position of the car. Because of [2], instead of extrapolating :math:`\bar{v}` into the future,
we can find its position by moving it along the circular path.

.. math::

    x_{next} = x + r_{rear}(sin(\psi_{next}) - sin(\psi)) + L_{rear}(cos(\psi_{next}) - cos(\psi))

    y_{next} = y - r_{rear}(cos(\psi_{next}) - cos(\psi)) + L_{rear}(sin(\psi_{next}) - sin(\psi))

We use :math:`r_{rear}` instead of :math:`r` because the car body is perpendicular not to the turning axis, but the line
connecting the rear wheel to the center of the turning circle.

The model is now complete.

.. math::

    \vec{f}(\vec{x}, \vec{u}, \Delta t) = \begin{bmatrix}
        x_{next} \\
        y_{next} \\
        \psi_{next} \\
        v_{next} \\
    \end{bmatrix} = \begin{bmatrix}
        x + r_{rear}(sin(\psi_{next}) - sin(\psi)) + L_{rear}(cos(\psi_{next}) - cos(\psi)) \\
        y - r_{rear}(cos(\psi_{next}) - cos(\psi)) + L_{rear}(sin(\psi_{next}) - sin(\psi)) \\
        \psi + \omega \Delta t \\
        | \vec{v} + \delta v \Delta t |
    \end{bmatrix}

.. no slip = no friction force to account for?

.. Newton's method?

.. All you need is swnagle and speed

