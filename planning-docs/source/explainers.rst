Explainers
=============

How is Midline Generated?
---------------------------
Our midline calculation relies on Support Vector Machines (:term:`SVM`\ s). Using the cones received from our lidar, the :term:`SVM` generates a decision boundary using a cubic polynomial kernel. This decision boundary is converted into a series of points that represents the path the car should take.

Why :term:`iSAM2`?
-------------------
We chose this algorithm after considering 1. :term:`iSAM2` is capable of incrementally optimizing its estimates for previous cone position estimates and :term:`pose` estimates, as opposed to optimizing only at loop closure, 2. :term:`iSAM2`'s performance made it a clear choice for :term:`SLAM`, and 3. we are able to work closely with the author of :term:`iSAM2`, Professor Michael Kaess. Thus, we would like to take this opportunity to thank Professor Michael Kaess for dedicating his time and efforts to assisting our implementation of :term:`SLAM`.

What is :term:`iSAM2` and :term:`Factor Graph` :term:`SLAM`?
---------------------------------------------------------------

:term:`iSAM2` (Incremental Smoothing and Mapping) is a :term:`SLAM` (Simultaneous Localization and Mapping) algorithm used to construct a map of the track from the car's position and the cones observed around the track. :term:`iSAM2` does this by constructing and optimizing a :term:`Factor Graph` containing :term:`Variable Node` s (which represent either :term:`landmark` poses or car poses) and :term:`Factor Node` s.

The :term:`Factor Graph`
-------------------------
As stated previously, :term:`iSAM2` relies on a :term:`Factor Graph` containing :term:`Variable Node`\ s and :term:`Factor Node`\ s. :term:`Factor Node`\ s, akin to labeled edges between the :term:`Variable Node`\ s, represent a :term:`joint probability distribution` on the :term:`Variable Node`\ s connected to it. This :term:`joint probability distribution` represents how certain :term:`iSAM2` is of the corresponding variables' positions.

.. image:: ./img/factor_graph.png
    :align: center

.. note:: A :term:`Variable Node` cannot be adjacent to another :term:`Variable Node` and a :term:`Factor Node` cannot be adjacent to another :term:`Factor Node`. Blue nodes represent :term:`Variable Node`\ s (X variable nodes represent car poses, L variable nodes represent :term:`landmark` positions). Gray nodes represent :term:`Factor Node`\ s. :term:`Factor Node` :math:`f_{0}` is called a :term:`Prior Factor` node; :term:`prior factor` nodes are added to the first :term:`pose` and sometimes the first :term:`landmark` for :term:`iSAM2` to use as reference when localizing and mapping future :term:`pose`\ s and :term:`landmark`\ s.

For example, observe how in Figure 1, :term:`Factor Node` :math:`f_{1}` is connected to :term:`Variable Node` :math:`x_{0}`, representing the first car :term:`pose`, and :math:`l_{0}`, representing the first :term:`landmark`. :term:`Factor Node` :math:`f_{1}` represents a :term:`joint probabilistic distribution` function over :math:`x_{0}` and :math:`l_{0}`, which indicates how certain :term:`iSAM2` is of the positions for :math:`x_{0}` and :math:`l_{0}`. Altogether, the entire :term:`Factor Graph` represents a :term:`joint probabilistic distribution` function F on all :term:`landmark` positions and car :term:`pose`\ s. This function F is equal to the product of all factors :math:`f_{n}`, the :term:`joint probabilistic distribution` function represented by each :term:`Factor Node` in the graph.

**Goal with respect to the Factor Graph**
The goal of :term:`iSAM2` is to maximize the :term:`joint probabilistic distribution` function F by maximizing its factors. Intuitively, :term:`iSAM2` is seeking to maximize its certainty of :term:`landmark` positions and car :term:`pose`\ s by updating its estimates for car :term:`pose`\ s and :term:`landmark` positions over time (with the help of incoming observations). Considering the previous example, :term:`iSAM2` can maximize this function F by maximizing the :term:`joint probability distribution` function represented by :term:`Factor Node` :math:`f_{1}`.

**Implementation**

.. image:: ./img/observation_step.png
    :align: center

The :term:`iSAM2` node first parses the cones received by perceptions into separate vectors by color. This vector of observed cones and other :term:`Odometry` information is used to update the :term:`iSAM2` model. Using the :term:`Odometry` information, the :term:`iSAM2` node predicts the car's current :term:`pose` using the received :term:`Odometry` information. :term:`Variable Node` :math:`x_{n}`, representing the car :term:`pose` at the current time stamp, is added alongside a :term:`Factor Node` connecting :math:`x_{n}` to :math:`x_{n-1}`, the :term:`Variable Node` representing the previous car :term:`pose`.

.. image:: ./img/data_association.png
    :align: center

After determining the car :term:`pose`, :term:`Data Association` is performed on the cones observed at the current timestamp to determine which of the observed cones are new. To perform this :term:`Data Association`, the :term:`Mahalanobis Distance` is calculated between one observed cone, and all :term:`iSAM2` estimates for the previously seen cones. Intuitively, the :term:`Mahalanobis Distance` represents how much the observed cone resembles a previously seen cone (the smaller the distance, the more the observed cone resembles the previously seen cone). If the smallest distance is greater than the :term:`Mahalanobis Distance` Threshold, then the observed cone is a new cone.

.. note:: The :term:`Mahalanobis Distance` threshold is generally found through tuning and trial and error.

.. note:: :term:`Mahalanobis Distance` is used instead of Euclidean distance because where Euclidean distance can calculate the distance between two points, :term:`Mahalanobis Distance` can calculate the distance between a point and a distribution. This is important because the cone positions come with uncertainty which is represented by a distribution (See `more`_)

.. _more: https://www.machinelearningplus.com/statistics/mahalanobis-distance/

.. image:: ./img/updated_factor_graph.png
    :align: center

This process is repeated for all observed cones. Each detected new cone must be added to the :term:`Factor Graph` as a :term:`Variable Node` with a :term:`Factor Node` connected to :math:`x_{n}`, the :term:`Variable Node` representing the current car :term:`pose`.
