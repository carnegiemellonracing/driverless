#pragma once

#include <bits/stdc++.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose2.h>
/*
 * @brief This is a factor used for localizing car poses.
 * Unary factors are added to car poses. Car poses
 * are calculated by the velocity motion model, unary
 * factors are contain the GPS measurments
 */
class UnaryFactor : public gtsam::NoiseModelFactor1<gtsam::Pose2> {
    // x y measurements from GPS
    gtsam::Pose2 z;
    gtsam::Key key;

public:
    typedef std::shared_ptr<UnaryFactor> shared_ptr;

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
     * Below is the constructor
     */
    UnaryFactor(gtsam::Key k, const gtsam::Pose2& input_GPS, const gtsam::SharedNoiseModel& model):
        NoiseModelFactor1<gtsam::Pose2>(model, k), z(input_GPS.x(), input_GPS.y(), input_GPS.theta()), key(k) {}


    virtual ~UnaryFactor() {}

    /* @brief Evaluates the error between an unknown variable and the
     * GPS measurement we receive
     */
    gtsam::Vector evaluateError(const gtsam::Pose2& q, gtsam::OptionalMatrixType H) const override;
    /*
     * @brief Clones this factor
     */
    gtsam::NonlinearFactor::shared_ptr clone() const override {
    	return std::make_shared<UnaryFactor>(this->key, z, this->noiseModel_);
    }
};


