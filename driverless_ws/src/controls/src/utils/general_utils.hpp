#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <glm/glm.hpp>

namespace controls {
    /// Asserts whether x is true if the PARANOID compiler flag is set (-P)
    #ifdef PARANOID
    #define paranoid_assert(x) (assert(x))
    #else
    #define paranoid_assert(x) ((void)0)
    #endif

    struct SplineAndCones
    {
        std::vector<glm::fvec2> spline;
        std::vector<glm::fvec2> left_cones;
        std::vector<glm::fvec2> right_cones;
    };

    inline std::string points_to_string(std::vector<glm::fvec2> points)
    {
        std::stringstream ss;
        for (size_t i = 0; i < points.size(); i++)
        {
            ss << "[" << points.at(i).x << "," << points.at(i).y << "]";
            if (i < points.size() - 1)
            {
                ss << ", ";
            }
        }
        return ss.str();
    }

    /// Rotates a point by an angle
    inline glm::fvec2 rotate_point(glm::fvec2 point, float angle) {
        return {
            glm::cos(angle) * point.x - glm::sin(angle) * point.y, 
            glm::sin(angle) * point.x + glm::cos(angle) * point.y
            };
    }

    inline bool isnan_vec(glm::fvec2 vec) {
        return std::isnan(vec.x) || std::isnan(vec.y);
    }
}

