/** 
 * Controller implementation 
 * 
 * @author Griffin Teller <gteller@andrew.cmu.edu>
*/

#include "controller.hpp"
#include "model.hpp"
#include <cmath>
#include <exception>
#include <cassert>

namespace controls {

    int main(int argc, char *argv[]) {
        rclcpp::init(argc, argv);

        auto pControllerNode = std::make_shared<ControllerNode>();
        rclcpp::spin(pControllerNode);

        rclcpp::shutdown();
        return EXIT_SUCCESS;
    }



    // ***** CONTROLLER NODE ******

    ControllerNode::ControllerNode() {
        /* Creates controller node and initializes spline subscription, but 
           waits for first spline to start optimization threads and 
           publishers. */

        m_subscription = create_subscription<SplineMsg>(
            SPLINE_TOPIC_NAME, 10, [=] (auto msg) {
                splineCallback(msg);
            }
        );

        // needs to be initialized in case first spline comes before first timer
        // callback
        m_speedPid.recordState(m_vehicleState.x_dot, SLOW_LAP_SPEED, now());
    }

    void ControllerNode::timerCallback() {
        calculateVehicleState();
        if (isSlowLap()) {
            m_speedPid.recordState(m_vehicleState.x_dot, SLOW_LAP_SPEED, now());
        }

        ControlAction action = m_lastAction;

        if (m_torquePlanningFuture.valid) {
            if (m_torquePlanningFuture.wait_for(0) == future_status::ready) {
                auto torques = m_torquePlanningFuture.get();
                std::copy(torques.begin(), torques.end(),
                          &action.data[1]);
            } else {
                RCPCPP_INFO_ONCE("Controller received timer but torque planning"
                                 " is still running. Sending last torque...");
            }
        } else {
            RCPCPP_INFO_ONCE("Controller received timer but torque planning "
                             "is not active. Sending last torque...");
        }

        // TODO: implement control action I/O
        throw std::runtime_error("control action i/o not implemented");
    }

    void ControllerNode::splineCallback(SplineMsg::SharedPtr msg) {
        /* Update reference spline and start torque planning thread, if it is
           not active */

        if (m_torquePlanningFuture.valid()) {
            RCPCPP_INFO_ONCE("Controller received spline but torque planning "
                             "has not finished. Ignoring...");
            return;
        }

        m_pReferenceSpline = std::make_unique<ReferenceSpline>(*msg);
        if (m_waitingForFirstSpline) {
            m_waitingForFirstSpline = false;
        }

        m_torquePlanningFuture = std::async([=] () {
            return calculateTorques();
        })
    }

    double ControllerNode::calculateSteering() const {
        /* à la https://arxiv.org/pdf/2111.08873.pdf, among other resources.
           Assumptions: Vehicle is always on spline. Car always has heading 
           0. */

        const double lookahead = getLookahead();
        const ReferenceSpline::Frame lookaheadSplineFrame = 
            m_pReferenceSpline->poseAtDistance(lookahead);

        const double headingDeviation = lookaheadSplineFrame.tangentRad;
        const lookaheadStraightDistance = sqrt(
            lookaheadSplineFrame.x * lookaheadSplineFrame.x
          + lookaheadSplineFrame.y * lookaheadSplineFrame.y
        );
        const discSub = 2 * REAR_BASE_FRAC * WHEELBASE * sin(headingDeviation);

        const lookaheadStraightDistance2 = 
            lookaheadStraightDistance * lookaheadStraightDistance;
        const discSub2 = discSub * discSub;

        return atan(
            2 * WHEELBASE * sin(headingDeviation) / 
            sqrt(lookaheadStraightDistance2 - discSub2)
        );
    }

    std::array<double, N_TIRES>
    ControllerNode::calculateTorques(VehicleState vehicleState) const {
        double totalTorque;
        if (isSlowLap) {
            totalTorque = m_speedPid.getAction();
        } else {
            const double distToCorner
        }

        std::array<double, N_TIRES> res;
        for (int i = 0; i < N_TIRES; i++) {
            res[i] = totalTorque / N_TIRES;
        }
        return res;
    }

    void ControllerNode::calculateVehicleState() {
        // TODO: figure out gps msg type and implement vehicle state parsing
        throw new std::runtime_error("vehicle state parsing not implemented");
    }

    double ControllerNode::getLookahead() const {
        return m_vehicleState.x_dot * LOOKAHEAD_WEIGHT + LOOKAHEAD_BIAS;
    }

    bool ControllerNode::isSlowLap() const {
        // TODO: determine how to retrieve lap info
        throw new std::runtime_error("lap info parsing not implemented");
    }



    // ***** PID *****

    PIDController1D::PIDController1D(double kp, double ki, double kd)
    : m_kp(kp), m_ki(ki), m_kd(kd) {}

    void PIDController1D::setCoefficients(double kp, double ki, double kd) {
        m_kp = kp;
        m_ki = ki;
        m_kd = kd;
    }

    void PIDController1D::recordState(double state, double setpoint,
                                      rclcpp:Time time) {
        if (!m_init) {
            m_lastTime = time - rclcpp::Duration(PID_INIT_TIME);
            m_lastError = setpoint - state;
        } else {
            m_lastTime = m_currTime;
            m_lastError = m_currError;
        }

        m_currTime = time;
        m_currError = setpoint - state;

        double deltaT = (m_currTime - m_lastTime).seconds();
        m_errorIntegral += (m_currError + m_lastError) / 2 * deltaT;
        m_errorDerivative = (m_currError - m_lastError) / deltaT;
    }

    double PIDController1D::getAction() const {
        assert(m_init);

        return m_kp * m_currError
             + m_ki * m_errorIntegral
             + m_kd * m_errorDerivative;
    }

    void PIDController1D::clear() {
        m_init = false;
    }
}