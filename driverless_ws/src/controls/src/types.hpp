#pragma once

#include <state/state_estimator.hpp>
#include <array>

#include "constants.hpp"


namespace controls {
    using Action = std::array<float, action_dims>;
    using State = std::array<float, state_dims>;

    union SplineFrame {
        float4 texel;

        struct {
            float x;
            float y;
            float tangent_angle;
            float curvature;
        };
    };

    using SplineMsg = std::vector<SplineFrame>;
    using SlamMsg = struct {};

    class Controller {
    public:
        virtual Action generate_action() =0;

        virtual ~Controller() =0;
    };
}
