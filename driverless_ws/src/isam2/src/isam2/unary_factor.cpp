#include "unary_factor.hpp"
/*
 * @brief This is a factor used for localizing car poses.
 * Unary factors are added to car poses. Car poses
 * are calculated by the velocity motion model, unary
 * factors are contain the GPS measurments
 */

/*
 * In the iSAM2 paper, each factor f(q) is proportional to
 * exp(-||h(q) - z_{t}||^2)
 *
 * q is the unknown variable
 * z_{t} is our measurement (GPS in this case) at time t
 * - Here we represented this as x and y.
 *   (gtsam Factor Graph tutorial)
 *
 * This function represents our likelihood of q given measurement z
 * - We want to maximize this likelihood by minimizing the squared error
 *   In a sense, this function represents the error between an unknown
 *   variable and the measurement we receive
 *
 * h(q) is a nonlinear measurement function (gtsam Factor Graph tutorial)
 * - in our case we just return the variable itself (identity)
 *
 * What is q in the context of our problem: q is either a pose or a landmark
 * - (a variable node)
 * 
 */



/* @brief Evaluates the error between an unknown variable and the
 * GPS measurement we receive
 */
gtsam::Vector UnaryFactor::evaluateError(const gtsam::Pose2& q, gtsam::OptionalMatrixType H) const {
    const gtsam::Rot2& R = q.rotation(); // TODO: Probably do not need & in the type annotation?

    // If H is not none
    if (H) {
        (*H) = (gtsam::Matrix(2, 3) << R.c(), -R.s(), 0.0,
                                        R.s(), R.c(), 0.0).finished();
    }

    return (gtsam::Vector(2) << q.x() - z.x(),
                                q.y() - z.y()).finished();
}






