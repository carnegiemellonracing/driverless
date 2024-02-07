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

        SplineFrame(float x, float y, float tangent_angle, float curvature)
            : x {x}, y {y}, tangent_angle {tangent_angle}, curvature {curvature} { }
    };

}