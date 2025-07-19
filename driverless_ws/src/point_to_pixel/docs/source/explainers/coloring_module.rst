Coloring Module Concepts
========================

Once we have finished the processing in the :doc:`LiDAR module <../explainers/lidar_module>`, 
we are left with a set of 3d coordinates representing cones in space. However, our path planning 
and controls modules expect these points to be colored as well. For that, we use our cameras to provide the 
color information needed to classify cones.

Our method of sensor fusion is relatively simple. We make the assumption that cameras remain rigid
compared to the LiDARs at all time and can simply use the static transformation matrix from camera to LiDAR. 
We use a technique known as the `direct linear transform  <https://en.wikipedia.org/wiki/Direct_linear_transformation>`_
to calculate such a matrix. Finally, we use our point to pixel mapping library, which leverages that transform
and a custom YOLO v5 model to classify cone colors.


:doc:`Direct Linear Transform <>`
-------------------------------------------------------------------

Why?
^^^^
We cannot rely on CAD to calculte the transform matrix as there are always slight differences in parts from CAD to manufacturing.
This transform matrix has to be extremely accurate in order to properly classify images.

What?
^^^^^
Instead, we solve for the transform matrix via a calibration secquence that uses a series of (at least 6) points identified by 
hand in both the lidar and camera frames. With those points we can solve for the matrix 

.. figure:: ppm_calibration.JPG
    :width: 600
    :align: center

    *Figure: calibration setup consisting of many cones spread throughout at different heights and depths*

.. Warning::
    This method is heavily dependant on a good calibration. If the sensors move relative to each other 
    or if the calibration points weren't picked at various depths / heights, the accuracy drops off steeply.

.. Note::
    - Add diagram + equations for DLT


:doc:`Point to Pixel Mapping <../implementation/coloring_module>`
-------------------------------------------------------------------

.. Note::
    - PPM system diagram

.. figure:: ppm_system_diagram.png
    :width: 600
    :align: center


With transform calculated, we can run our point to pixel sensor fusion algorithm.


Simplified Algorithm
^^^^^^^^^^^^^^^^^^^^
.. code-block:: text

    for each cone centroids / camera image do:
        YOLO v5 cone detection inference: bounding boxes of different color/size classes

        For each cone centroid do:
            transform to image space
            
            if point is within a single bounding box classify as that color

            if point is in multiple boxes use a rough depth heuristic to pick one box

            else label the point as unkown

Notes
"""""
- Our YOLO v5 is trained on data from `Formula Student Objects in Context Dataset (FSOCO) <https://fsoco.github.io/fsoco-dataset/>`_
- Depth heuristic uses the idea that the area of the bounding box roughly corresponds to depth 

Complexities related to sensor frame timestamps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main complexity in our implementation is to solve two problems that crop up as a result of unsynchronized data.

    1. Our sensors have no synchronous trigger, and thus we need to find a way to syncronize data as best as possible.
    2. Our system needs to be robust to a a few errors in coloring due to this LiDAR - camera temporal synchronization issue.

This adds some complexity to the algorithm:

.. code-block:: text

    for each incoming set of cone centroids do: 
        find the camera image in buffer closes to timestamp of centroids (but after)
        
        apply YOLO v5 cone detection to that image: bounding boxes of different color/size classes

        For each cone centroid do:
            motion model point based on velocity and yaw deltas between LiDAR centroid timestamp and image timestamp

            transform to image space
                
            if point is within a single bounding box classify as that color

            if point is in multiple boxes use a rough depth heuristic to pick one box

            else label the point as unkown

        feed centroids into cone history algorithm to minimize the effect of misclassifications

        apply support vector machine to correct for up to 1-2 misclassifications


Notes
"""""

- Our Movella IMU is used to get velocity and yaw deltas.
- Cone histories and SVM make the algorithm far more robust to synchronization issues--especially at faster speeds.
