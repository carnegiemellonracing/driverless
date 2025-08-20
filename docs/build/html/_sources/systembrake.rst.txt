Autonomous System Brake (ASB)
==================================

The ASB is responsible for slowing the vehicle to a stop. It includes an Emergency Braking System (EBS) to stop the vehicle in the case of a safety/power failure or if the emergency stop RES is hit.

Rules Overview:
------------------

Below are rules from the 2025 FSAE Driverless Supplement that constrained vehicle retrofitting solutions for ASB.

**DT.3.1 Driverless System Brake**

The vehicle must have a Driverless System Brake. The Tractive System is not a Brake System 

DT.3.1.1 Technical Requirements 

a. All parts and their mountings must be located inside the Rollover Protection Envelope 

b. Manual braking must always be possible and not prevented at any time 

c. The DSB may be part of the hydraulic brake system 

DT.3.1.1 defines key limits for ASB including the mounting area and requirement that manual braking must to always be possible.

DT.3.1.2 DT.3.1.1 defines key requirements for ASB 

a. The Driverless System Brake must be designed to be deactivated by a maximum of two Deactivation Points 

b. The Deactivation Points must be:
    - Mounted inside the volume defined in DT.2.3.a 
    - Mounted in one of the two: near the DSMS or on the top side of the vehicle between the Front Bulkhead and Front Hoop close to the vehicle center line 
    - Near each other 
    - Protected against unintended actuation (being hit by a cone) while driving 
    - Marked with “Brake Release” 
    - Fitted with a red handle 
    - Able to work without electrical power 
    - Operated by maximum two simple push/pull and/or turning actions, the order and direction of these actions must be shown next to the Deactivation Points 

DT.3.1.2 defines key requirements for the deactivation points for the ASB release including mounting location and basic requirements. 

**DT.3.2 Emergency Brake System (EBS)**

The Driverless System Brake must include an Emergency Brake System (EBS)

DT.3.2.1 Technical Requirements 

a. The EBS must only use passive systems with mechanical energy storage 

b. The EBS must be directly supplied by GLVMS, DSMS, Remote Stop Relay and the Emergency Brake System Relay with no delay 

DT.3.2 defines key requirements for the Emergency Brake System (EBS) including that it must use stored mechanical energy and trigger in an emergency state. 

DT.3.2.2 Function 

a. Startup Check - must be performed to ensure that DSB is able to build up brake pressure  as expected, before Driverless System Status Ready is possible 

b. After the Startup Check the DSB and its signals must be continuously monitored for failures 

c. Electrical power loss at EBS must start an Emergency Brake Maneuver

DT.3.2.3 The vehicle must go to the Safe State, if:

a. Functionality of the Emergency Brake System cannot be ensured 

b. An (additional) single point of failure would lead to total loss of brake capability DT.3.2.4 The Safe State is when the three:  

a. Vehicle at a standstill

b. Brakes engaged to prevent the vehicle from rolling 

c. An open Shutdown Circuit  

DT.3.2.2-4 defines the function and state behavior required by EBS.

DT.3.2.5 Emergency Brake Maneuver 

The Emergency Brake System must decelerate the vehicle and stop vehicle motion 

a. The system reaction time, the time between opening of the Shutdown Circuit and the  start of the deceleration, must be 200 ms or less 

b. The average deceleration must be more than 10 m/s2 under dry track conditions 

c. In case of a single failure the DSB should achieve at least half of the performance 

d. While decelerating, the vehicle must remain in a stable driving condition 

DT.3.2.5 defines the base system requirements for EBS. 

System Design:
----------------

.. figure:: ./img/asbdesign.png
    :align: center

    FSAE Steering Diagram `(reference) <https://medium.com/@luisdamed/brake-system-load-distribution-study-matlab-approach-2f35b426ee0d>`_

FSAE Brakes Diagram (reference)

FSAE brakes begin with two master cylinders. These master cylinders are mounted on the manual brake pedal. When the driver pushes the pedal down, the master cylinders compress. Additionally, each wheel on the vehicle has brake calipers mounted on its hub, these connect hydraulically to the master cylinders. When the master cylinders are compressed, brake fluid pushes to the calipers, compressing the brake pads to lock the wheels. 

The main requirement of Autonomous System Brake (ASB) can be fulfilled by regeneration braking. The motors are sent negative torque commands which are used to decelerate the car, serving as a driverless form of braking. 

The addition of the EBS required by the rules must utilize mechanical energy and not inhibit the compression of the brake pedal manually. There are two main design approaches to EBS:

1. Hydraulic integration with the existing brake lines

    - Addition of two driverless-specific master cylinders that hydraulically interface with the pedal master cylinder through 3-way valves allowing the side with the higher pressure to flow to the outlet. The line with the lower pressure is sealed off, allowing for pressure to build up and lock the calipers based on which master cylinder is currently compressed.

2. Physical actuation of the pedal

    - Actuation of the pedal involving pneumatics or other mechanically stored force to physically move the brake pedal and compress the existing pedal master cylinders. 

After EBS is triggered, the braking must be released via manual release valves. Per the rules these must be mounted on the top face of the vehicle near the center line such that anyone can release brakes. 

Implementation:
------------------

Regular braking throughout a driverless run fulfilled by regenerative braking.

*EBS*

This implementation of the emergency braking system involves a fully redundant pneumatic and hydraulic system. Two additional master cylinders were added to the existing braking system and were autonomously actuated. These fluid lines interface with the manual master cylinder fluid lines at 3-way valves which allow dual manual and autonomous braking. As it is redundant, if one part of this system fails, 50% of the braking functionality will still apply. 

.. figure:: ./img/brakesschematic.png
    :align: center

    Carnegie Mellon Racing Driverless and Manual Brakes Integration Schematic

1. Shared Pneumatic Supply

    a. Power Supply - Large Air Tank

    b. Pressure Regulator - 800 PSI Output

    c. Pressure Regulator - 150 PSI Output

    d. Over Protection - 140 PSI Output

    e. Tee Fitting - Splits into separate front and rear braking lines for redundance

2. Separate Pneumatic Systems

    a. Check Valves - Ensures no back flow from holding chambers to power supply

    b. Holding Chambers  - Holds air to ensure redundancy if pressure supply fails

    c. Manual Release Valves - Remove air from one side of the line

    d. Pneumatic Pressure Sensors - Allows for continuous monitoring of pressure within the lines

    e. Normally Open Solenoids - Allows air travel when the vehicle has no power.

    f. Pneumatic Pistons - When pressurized, these actuate directly on the EBS master cylinders

    g. EBS Master Cylinder - Pushes brake fluid down to 3-way valve

3. Separate Hydraulic Lines

    a. 3-way valve - Fluid from both the EBS and Manual Master Cylinders meet. The highest pressure flows through

    b. Brake Caliper - Outlet of 3-way valve directs brake fluid to the calipers which lock the wheels to stop the vehicle 

.. figure:: ./img/ebspneumatic.png
    :align: center

    EBS Pneumatic and Master Cylinder Interface Model

The EBS master cylinders are compressed by the pneumatic cylinders which are triggered by normally open solenoids. 

The brake lines from the EBS master cylinders meet with the brake lines from the manual master cylinders mounted on pedals at a 3-way valve. The 3-way valve allows the input with the highest pressure flow through to the output. This enables the switch between EBS and manual braking based on which side has a higher pressure. 

The air is exhausted from the system through a pair of 3-way manual release valves mounted on the dashboard near the center line of the car in alignment with rule specifications.
