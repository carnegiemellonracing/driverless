Class state::StateEstimator_Impl
==========================================

Prepare yourself.

State Projection
----------------
Done in a separate object called the StateProjector which is owned by StateEstimator_Impl. This is a brief description.

The main data structure is a multiset, which is essentially a binary search tree. We use it as a self-sorting array.

Spline and Twist ROS messages, along with computed and taken Control Actions, are mutated and placed into the multiset
as "Records".

- Spline message -> (0,0,0) pose with timestamp
- Twist message -> calculate norm (speed), add timestamp
- Control Action -> unmodified, timestamp + ``approx_propagation_delay`` added to estimate when the actuation **actually**
ocurs.

Records before the most recent spline are discarded because they exist in a different coordinate frame.

On a call to project(), project from each record in the multiset, using each record to make corrections to state or action.

Curvilinear Lookup Table Generation
-----------------------------------






.. doxygenclass:: controls::state::StateEstimator_Impl
