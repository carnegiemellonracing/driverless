#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <glm/glm.hpp>
#include <stdexcept>
#include "paranoid_assert.h"

namespace controls {
    /// Asserts whether x is true if the PARANOID compiler flag is set (-P)

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

    class ControllerError : public std::runtime_error {
        public:
            using std::runtime_error::runtime_error;
    };
    inline constexpr float degrees_to_radians(const float degrees) {
        return degrees * M_PI / 180.0f;
    }

    inline constexpr float radians_to_degrees(const float radians) {
        return radians * 180.0f / M_PI;
    }

    inline std::string vector_to_parseable_string(const std::vector<glm::fvec2>& vec) {
            if (vec.size() == 0) {
                return "EmptyVector";
            }
            std::stringstream ss;
            for (int i = 0; i < vec.size() - 1; i++)
            {
                ss << vec[i].x << " " << vec[i].y << ",";
            }
            ss << vec[vec.size() - 1].x << " " << vec[vec.size() - 1].y;

            return ss.str();
    }


}

