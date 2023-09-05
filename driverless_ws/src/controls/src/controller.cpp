/** 
 * Controller implementation
*/

#include "controller.hpp"
#include "model.hpp"
#include <cmath>
#include <exception>
#include <cassert>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix>

using std::placeholders::_1;

namespace controls {

    int main(int argc, char *argv[]) {
        rclcpp::init(argc, argv);

        auto pControllerNode = std::make_shared<ControllerNode>();
        rclcpp::spin(pControllerNode);

        rclcpp::shutdown();
        return EXIT_SUCCESS;
    }


    // ***** CONTROLLER NODE ******

    ControllerNode::ControllerNode()
        : Node(NODE_NAME) {
        /* Creates controller node and initializes spline subscription, but 
           waits for first spline to start optimization threads and 
           publishers. */

        m_splineSubscription = create_subscription<SplineMsg>(
            SPLINE_TOPIC_NAME, 10, std::bind(&ControllerNode::splineCallback, this, _1)
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

        if (m_torquePlanningFuture.valid()) {
            if (m_torquePlanningFuture.wait_for(0ms) == std::future_status::ready) {
                auto torques = m_torquePlanningFuture.get();
                std::copy(torques.begin(), torques.end(),
                          &action.data[1]);
            } else {
                RCLCPP_INFO_ONCE(get_logger(), "Controller received timer but"
                                               " torque planning is still"
                                               " running. Sending last torque...");
            }
        } else {
            RCLCPP_INFO_ONCE(get_logger(), "Controller received timer but torque"
                                           " planning is not active. Sending"
                                           " last torque...");
        }

        // TODO: implement control action I/O
        throw std::runtime_error("control action i/o not implemented");
    }

    void ControllerNode::splineCallback(const SplineMsg::SharedPtr msg) {
        /* Update reference spline and start torque planning thread, if it is
           not active */

        if (m_torquePlanningFuture.valid()) {
            RCLCPP_INFO_ONCE(get_logger(), "Controller received spline but"
                                           " torque planning has not finished."
                                           " Ignoring...");
            return;
        }

        m_pReferenceSpline = std::make_unique<ReferenceSpline>(*msg);
        if (m_waitingForFirstSpline) {
            m_waitingForFirstSpline = false;
        }

        m_torquePlanningFuture = std::async([=] () {
            return calculateTorques();
        });
    }

    double ControllerNode::calculateSteering() const {
        /* à la https://arxiv.org/pdf/2111.08873.pdf, among other resources.
           Assumptions: Vehicle is always on spline. Car always has heading 
           0. */

        const double lookahead = getLookahead();
        const ReferenceSpline::Frame lookaheadSplineFrame = 
            m_pReferenceSpline->poseAtDistance(lookahead);

        const double headingDeviation = lookaheadSplineFrame.tangentRad;
        const double lookaheadStraightDistance = sqrt(
            lookaheadSplineFrame.x * lookaheadSplineFrame.x
          + lookaheadSplineFrame.y * lookaheadSplineFrame.y
        );
        const double discSub = 2 * REAR_BASE_FRAC * WHEELBASE * sin(headingDeviation);

        const double lookaheadStraightDistance2 =
            lookaheadStraightDistance * lookaheadStraightDistance;
        const double discSub2 = discSub * discSub;

        return atan(
            2 * WHEELBASE * sin(headingDeviation) / 
            sqrt(lookaheadStraightDistance2 - discSub2)
        );
    }

    std::array<double, N_TIRES> ControllerNode::calculateTorques() {
        double totalTorque = isSlowLap() ?
                m_speedPid.getAction() : getFastLapTorque();

        std::array<double, N_TIRES> res;
        for (uint i = 0; i < N_TIRES; i++) {
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

    double ControllerNode::getFastLapTorque() {
        const double progressTillCorner = findNearestCorner();

        if (progressTillCorner >= m_lastCornerDistance
                                + SEGMENT_CHANGE_TOLERANCE
         && m_lastCornerDistance < MIN_SEGMENT_CHANGE_CORNER_DIST) {
            m_accelerationPhase = true;
        }
        m_lastCornerDistance = progressTillCorner;

        const ReferenceSpline::Frame cornerFrame =
                m_pReferenceSpline->poseAtDistance(progressTillCorner);

        const double currSpeed = m_vehicleState.x_dot;
        const double desiredCornerSpeed = m_ggv.getTractiveCapability(
                                                currSpeed,
                                                cornerFrame.curvature);

        const double requiredDeceleration =
                (currSpeed * currSpeed
                - desiredCornerSpeed * desiredCornerSpeed)
                / (2 * progressTillCorner);

        if (m_accelerationPhase) {
            if (requiredDeceleration < TARGET_BRAKE) {
                return TARGET_ACCEL;
            } else {
                m_accelerationPhase = false;
            }
        }

        m_brakePid.recordState(requiredDeceleration, TARGET_BRAKE, now());
        return -m_brakePid.getAction();
    }

    double ControllerNode::findNearestCorner() const {
        // TODO: figure out how to find corners
        throw new std::runtime_error("corner finding not implemented");
    }

    void ControllerNode::loadGGV() {
        // Eventually, we will implement a real GGV which will need to be loaded
        // This function was declared just so we don't forget :P

        return;
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
                                      rclcpp::Time time) {
        if (!m_init) {
            m_lastTime = time - rclcpp::Duration(PID_INIT_TIME);
            m_lastError = setpoint - state;
            m_init = true;
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


    // ***** GGV *****

    double GGV::getTractiveCapability(double curvature) const {
        // TODO: Implement an actual GGV
        return sqrt(GRAVITY * TRACTIVE_DOGSHIT_COEF * LAT_MU / curvature);
    }


    // ***** REFERENCE SPLINE *****

    ReferenceSpline::ReferenceSpline(const SplineMsg& rSplineMsg) {
        m_splines = std::vector<Spline>();
        for (auto splineMsg : rSplineMsg.splines) {
            Spline spline {};

            // Set spl_poly
            polynomial poly {};
            poly.deg = POLY_DEG;
            poly.nums = gsl_vector_alloc(POLY_DEG + 1);

            for (uint i = 0; i <= POLY_DEG; i++) {
                gsl_vector_set(poly.nums, i, splineMsg.spl_poly_coefs[i]);
            }

            spline.set_SplPoly(poly);

            // Set points
            gsl_matrix *points = gsl_matrix_alloc(
                    splineMsg.points.height, splineMsg.points.width);


            // TODO: finish implementing spline deserialization
            throw new std::runtime_error("spline deserialization not implemented");
        }
    }
}