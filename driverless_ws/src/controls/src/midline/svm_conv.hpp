#include "cones.hpp"
#include <vector>

namespace controls {
    namespace midline {

        typedef std::vector<std::pair<double, double>> conesList;

        conesList cones_to_midline(Cones cones);
    }
}