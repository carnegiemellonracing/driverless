#include "../cones.hpp"
#include "svm.hpp"

namespace cones {
namespace recolouring {
    cones::TrackBounds recolour_cones(cones::TrackBounds track_bounds, double C);
    // double node_predictor(const std::vector<double> &cone, const svm_model *model);
} // namespace recolouring
} // namespace cones