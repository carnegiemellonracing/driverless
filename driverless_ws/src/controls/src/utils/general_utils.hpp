#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <glm/glm.hpp>

namespace controls {
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
}

