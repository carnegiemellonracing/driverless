#pragma once

namespace controls {

    union SplineFrame {
        float4 texel;

        struct {
            float x;
            float y;
            float tangent_angle;
            float curvature;
        };
    };

}