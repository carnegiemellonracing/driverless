#pragma once

#include "state_estimator.hpp"


namespace controls {
    namespace state {

        union SplineFrame {
            float2 texel;

            struct {
                float tangent_angle;
                float curvature;
            };
        };

    }
}