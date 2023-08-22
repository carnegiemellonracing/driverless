#pragma once

#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <chrono>
#include <mutex>
#include <future>
#include <cstddef>
#include <planning/raceline.hpp>

using namespace std::literals::chrono_literals;

namespace controls {
    /* ALL COORDINATES ARE X-FORWARD, Z-UP, RIGHT-HANDED */

    /**
         * Spline message type. Until this is determined, I'm using a string
         * type as a placeholder.
        */
    using SplineMsg = std_msgs::msg::String;

    /** Number of dimensions in vehicle state */
    constexpr uint VEHICLE_STATE_DIMS = 6;

    /** Number of dimensions in control action */
    constexpr uint CONTROL_ACTION_DIMS = 5;

    /** Number of tires, and consequently number of torque outputs */
    constexpr uint N_TIRES = 4;

    /** 
     * Configuration values. These will be moved to a configuration file 
     * eventually.
     */
    constexpr double KP_SPEED = 1, KI_SPEED = 1, KD_SPEED = 1;
    constexpr double KP_BRAKE = 1, KI_BRAKE = 1, KD_BRAKE = 1;
    constexpr const char* SPLINE_TOPIC_NAME = "spline_topic";
    constexpr const char* NODE_NAME = "controller";
    constexpr double SPEED_PROF_STEP = 1;
    constexpr double WHEELBASE = 2, REAR_BASE_FRAC = 0.5;

    // units: seconds, meters
    constexpr double LOOKAHEAD_WEIGHT = 0.2, LOOKAHEAD_BIAS = 1;

    constexpr double SLOW_LAP_SPEED = 5;
    constexpr auto PID_INIT_TIME = 0.1s;
    constexpr double TARGET_ACCEL = 7.5, TARGET_BRAKE = 7.5;  // m/s^2
    constexpr double MAX_BRAKE = 10.;  // m/s^2
    constexpr double LAT_MU = 1.;
    constexpr double TOTAL_MASS = 200.; // kg
    constexpr double GRAVITY = 9.81;

    /** Proportion of calculated capacity we try to attain */
    constexpr double TRACTIVE_DOGSHIT_COEF = 0.5;
    constexpr uint POLY_DEG = 3;


    /**
     * How close we have to have been to a corner for a distance increase to
     * signal a new segment
     */
    constexpr double MIN_SEGMENT_CHANGE_CORNER_DIST = 1.;  // m

    /**
     * How much the corner distance needs to jump to be sure we changed segments
     */
    constexpr double SEGMENT_CHANGE_TOLERANCE  = 2.; // m


    /** Entry point to controller node. Spawns node and performs cleanup. */
    int main(int argc, char *argv[]);

    /**
     * Vehicle state data structure. Contains curvilinear position info and 
     * car-relative velocity info. Curvilinear x is the distance of the car 
     * along the spline, and curvilinear y is the displacement from the spline.
     * Curvilinear yaw is the heading deviation from the spline tangent.
     *
     * Can be accessed by dimension name, or reinterpreted as an array by
     * accessing the `data` member.
     */
    struct VehicleState {
        union {
            struct {
                /** Distance (m) of vehicle along the spline */
                double x_curvilinear;

                /** 
                 * Signed perpendicular offset (m) of the vehicle from the 
                 * spline.
                 */
                double y_curvilinear;

                /** Signed heading difference (rad) from the spline tangent */
                double yaw_curvilinear;

                /** Car-relative longitudinal velocity (m/s) */
                double x_dot;

                /** Car-relative lateral velocity (m/s) */
                double y_dot;

                /** Angular velocity (rad/s) about z-axis */
                double yaw_dot;
            }

            /** 
             * All struct members reinterpreted as an array. This is a little
             * cleaner than a `reinterpret_cast` or address-of operation 
             * everytime passing as a vector is required.
             */
            double data[VEHICLE_STATE_DIMS];
        }
    }

    /**
     * Control action data structure. Contains steering wheel angle and torques
     * for all four tires. Torque magnitudes should be interpreted as the final
     * reactive torque applied by the ground, and with positive torque 
     * corresponding to forward acceleration.
     *    
     * Can be accessed by dimension name, or reinterpreted as an array by 
     * accessing the `data` member.
     */
    struct ControlAction {
        union {
            struct {
                /** Angle (rad) of steering rack (!= swangle) */
                double steeringAngle;

                /** Output torque of front-left tire */
                double torque_fl;

                /** Output torque of front-right tire */
                double torque_fr;

                /** Output torque of rear-left tire */
                double torque_rl;

                /** Output torque of rear-right tire */
                double torque_rr;
            }

            /** 
             * All struct members reinterpreted as an array. This is a little 
             * cleaner than a `reinterpret_cast` or address-of operation 
             * everytime passing as a vector is required.
             */
            double data[CONTROL_ACTION_DIMS];
        }
    }

    class GGV {
    public:
        double getTractiveCapability(double speed, double curvature) const;
    };

    /**
     * Spline representing desired trajectory of the vehicle. This type is 
     * INCOMPLETE, pending further discussions with path planning. It should
     * be capable of quickly returning the `Frame` of the spline at a 
     * particular arc length.
     */
    class ReferenceSpline {
    public:
        /**
         * Description of state at a particular point on spline. Contains point
         * (x, y), the heading in radians of the tangent, the left-normal, 
         * and the curvature. 
         */
        struct Frame {
            /** x-coordinate along spline, in meters */
            double x;

            /** y-coordinate along spline, in meters */
            double y;

            /** Heading in radians of tangent vector to spline */
            double tangentRad;

            /** 
             * Heading in radians of normal vector to spline. The normal vector
             * is always to the left of the tangent.
             */
            double normalRad;

            /** Signed curvature in meters^-1 */
            double curvature;
        };

        /** 
         * Constructs a ReferenceSpline from a reference to a path planning 
         * spline message.
         * 
         * @param[in] rSplineMsg Reference to path planning spline message
         */
        ReferenceSpline(const SplineMsg& rSplineMsg);

        /**
         * Determine `Frame` of spline at a distance along it. 
         * 
         * @param[in] distance Distance along spline
         * @return Frame indicating spline parameters at distance
         */
        Frame poseAtDistance(double distance);

        /**
         * Calculate total length of spline.
         * 
         * @return Total length of spline, in meters
         */
        double getLength() const;

    private:
        std::vector<Spline> m_splines;
    };

    /**
     * Basic PID controller over one variable. Records states, setpoints, and
     * times, and integrates and differentiates error. Resulting action can
     * be queried.
     */
    class PIDController1D {
    public:
        /** 
         * Construct controller with specific coefficients. Integration and
         * differentiation does not start until the first call to `recordState`.
         * 
         * @param[in] kp Proportional coefficient
         * @param[in] ki Integral coefficient
         * @param[in] kd Differential coefficient
         */
        PIDController1D(double kp, double ki, double kd);

        /** 
         * Set coefficients.
         * 
         * @param[in] kp Proportional coefficient
         * @param[in] ki Integral coefficient
         * @param[in] kd Differential coefficient
         */
        void setCoefficients(double kp, double ki, double kd);

        /**
         * Record state, setpoint, and time, and integrate and differentiate
         * error. Any calls to `getAction` depend on the most recent call to
         * this method.
         * 
         * @param[in] state Process state
         * @param[in] setpoint Process setpoint
         * @param[in] time Time of observation
         */
        void recordState(double state, double setpoint, rclcpp::Time time);

        /**
         * Compute action based on the three coefficients and stored data.
         * Can't be called until the first call to recordState.
         * 
         * @return Control action
         */
        double getAction() const;

        /**
         * Clears history. `getAction` will not be valid until `recordState` is
         * called.
         */
        void clear();

    private:
        double m_kp, m_ki, m_kd;

        /** Previous observation timestamp */
        rclcpp::Time m_lastTime;

        /** Most recent observation timestamp */
        rclcpp::Time m_currTime;

        /** Error of previous observation */
        double m_lastError;

        /** Error of most recent observation */
        double m_currError;

        /** Total error accumulation, calculated via trapezoidal sum. */
        double m_errorIntegral;

        /** Error time derivative */
        double m_errorDerivative;

        /** Whether first state has been recorded */
        bool m_init = false;
    }

    /**
     * 22a Controller Node. This node executes three main processes:
     * 
     *     - Pure Pursuit: 
     *          Calculates optimal steering wheel angle to guide the vehicle 
     *          to a setpoint ahead on the reference spline.
     *
     *     - Speed Control (slow lap only):
     *          Controls speed by calculating wheel torques derived from a
     *          simple PID  for the total torque across all wheels, which is
     *          biased evenly.
     *
     *     - Torque Planning (fast laps only):
     *          Runs in a separate thread, spawned every time a new spline is
     *          received. Accelerates until a predetermined amount of braking
     *          power would be required to safely enter the following corner,
     *          and then uses a PID to keep the expected speed through the
     *          corner at a safe speed, if the current brake action were to be
     *          maintained.
     */
    class ControllerNode : public rclcpp::Node {
    public:
        /**
         * Default construct the node. Sets up ros stuff and spawns speed 
         * profile generation thread.
         */
        ControllerNode();

    private:
        /**
         * Callback which publishes the control action. Keeping this on a timer
         * allows it to be used as a heartbeat for the node. 
         */
        void timerCallback();

        /**
         * Callback taking new spline information and updating ReferenceSpline
         */
        void splineCallback(SplineMsg::SharedPtr msg);

        /**
         * Perform pure pursuit steering calculations, and write the result
         * to `pSteeringAngle`.
         * 
         * @return Pure pursuit steering wheel
         */
        double calculateSteering() const;

        /**
         * Calculate tire torques by planning until next corner. Should be
         * called asynchronously. Passing in a copy of the vehicle state locks
         * it for the duration of calculation.
         * 
         * @return Torque for each tire
         */
        std::array<double, N_TIRES>
        calculateTorques(VehicleState vehicleState) const;

        /**
         * Calculate vehicle state based on stored reference spline and imu/gps
         * data, and store result in m_vehicleState.
         */
        void calculateVehicleState();

        /**
         * Generate the arc length lookahead (in meters) given the stored
         * reference spline and a particular vehicle state. Currently, this is 
         * just a function of spline-tangential speed.
         * 
         * @return Lookahead distance
         */
        double getLookahead() const;

        /**
         * Returns true if the car is currently completing the slow lap. This
         * determines whether or not the SLOW_LAP_SPEED is followed.
         *
         * @return True if slow lap, false otherwise
         */
        bool isSlowLap();

        /**
         * Generate the torque appropriate during a fast lap.
         *
         * If the average deceleration needed to reach the next corner at the
         * desired fraction of the maximum tractive capability is lower than
         * the target deceleration, than we request acceleration.
         *
         * Otherwise, a PID controller is used to control the deceleration of
         * the car such that the car enters the corner at the set speed.
         *
         * @return Total torque
         */
        double getFastLapTorque() const;

        /**
         * Find the nearest corner (maximal curvature point) on the stored
         * reference spline
         *
         * @return Track progress until the corner
         */
        double findNearestCorner() const;

        /** Load ggv from config */
        void loadGGV();

        /**
         * Whether or not the node is waiting for the first spline. If true,
         * no extra threads are running and no control actions are being
         * published.
        */
        bool m_waitingForFirstSpline = true;

        /**
         * Whether we have hit the brake point in the current segment (where the
         * target braking is TARGET_BRAKE). Only valid on fast laps.
         */
        bool m_accelerationPhase = true;

        /**
         * Last distance to corner. If this jumps up a huge amount from near 0,
         * we assume we've moved on to a new segment.
         */
        double m_lastCornerDistance = 0;

        /**
         * Future representing state of torque planning task
         */
        std::future<std::array<double, N_TIRES>> m_torquePlanningFuture;

        /**
         * Most recently calculated vehicle state. Recalculated every timer
         * callback or splineCallback.
         */
        VehicleState m_vehicleState;

        /**
         * Last control action sent. Used if a new control action cannot be
         * computed in time (e.g. slow torque planning)
         */
        ControlAction m_lastAction;

        /** PID for speed control during slow lap */
        PIDController1D m_speedPid {KD_SPEED, KI_SPEED, KP_SPEED};

        /**
         * PID for brake control during fast laps. Needs to be reinitialized
         * every segment.
         */
        PIDController1D m_brakePid {KD_BRAKE, KI_BRAKE, KP_BRAKE};

        /** 
         * Current reference spline. Updated with each call to 
         * splineCallback. Null if first spline has not been passed yet.
         */
        std::unique_ptr<ReferenceSpline> m_pReferenceSpline {nullptr};

        /** Subscriber to path planning spline */
        rclcpp::Subscription<SplineMsg>::SharedPtr m_splineSubscription;

        /** Performance envelope of vehicle */
        GGV m_ggv;
    }
}