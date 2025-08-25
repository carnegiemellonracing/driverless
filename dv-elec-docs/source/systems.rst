Driverless Electrical Systems
===============================

The driverless electrical subteam handles the integration of the necessary electrical systems that complies with the driverless regulatory requirements and allows the vehicle to operate autonomously. The following systems were retrofitted onto our existing FSAE vehicle, 24e, for the transition to a driverless FSAE vehicle, 24a: 

- Autonomous Interface Module (AIM)
- Autonomous Mission Indicator (AMI)
- Driverless System Status Indicator (DSSI)
- Driverless System Master Switch (DSMS)
- Remote Emergency Stop (RES)
- Emergency Braking System (EBS) Actuators
- Autonomous Steering System Actuators
- Onboard Computer and Perception Sensors

In addition to the hardware modifications, Driverless Electrical Systems is responsible for the necessary firmware changes to the vehicle's existing firmware stack to accommodate the newly added Driverless States: DS Ready, DS Driving, DS Finished, and DS Emergency, as well as the implementation of the Driverless inspection mission and safety check features to meet the driverless regulatory requirements.

24a System Block Design Diagram
------------------------------------
.. image:: ./img/24asystemblockdiagram.png
    :align: center

Driverless State Machine
---------------------------------
.. image:: ./img/driverlessstatemachine.png
    :align: center