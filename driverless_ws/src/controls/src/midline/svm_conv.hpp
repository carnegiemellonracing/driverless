#include "cones.hpp"
#include <vector>

namespace controls {
    namespace midline {

        typedef std::vector<std::pair<double, double>> conesList;
        namespace svm_fast  {

            conesList cones_to_midline(Cones cones);
        }

        namespace svm_slow  {

            conesList cones_to_midline(Cones cones);
        }
    }
}