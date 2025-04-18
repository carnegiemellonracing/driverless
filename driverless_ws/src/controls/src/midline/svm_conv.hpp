#include "cones.hpp"
#include <vector>
#include "svm.hpp"

namespace controls {
    namespace midline {

        typedef std::vector<std::pair<double, double>> conesList;
        namespace svm_fast  {

            conesList cones_to_midline(Cones cones);
        }

        namespace svm_slow  { 


            conesList cones_to_midline(Cones cones);
        }
        namespace svm_strat3{ //binary -> linear if fails
            conesList boundaryDetection(const std::vector<std::vector<double>> &xx, const std::vector<std::vector<double>> &yy,
                const svm_model *model);
            conesList cones_to_midline(Cones cones);
        }
        namespace svm_strat4{ //flood fill
            conesList boundaryDetection(const std::vector<std::vector<double>> &xx, const std::vector<std::vector<double>> &yy,
                const svm_model *model);
            conesList cones_to_midline(Cones cones);
        }
        namespace svm_naive{ //brute force
            conesList boundaryDetection(const std::vector<std::vector<double>> &xx, const std::vector<std::vector<double>> &yy,
                const svm_model *model);
            conesList cones_to_midline(Cones cones);
        }
        namespace svm_fast_double_binsearch{
            conesList boundaryDetection(const std::vector<std::vector<double>> &xx, const std::vector<std::vector<double>> &yy,
                const svm_model *model);
        }

        namespace svm_test {
            conesList cones_to_midline(Cones cones);

        }
    }
}