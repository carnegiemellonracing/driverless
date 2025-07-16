#pragma once
#include "../cones.hpp"
#include "../../../lib/svm_lib.hpp" // this is really shite way to do this

namespace cones {
namespace recoloring {
    cones::TrackBounds recolor_cones(cones::TrackBounds track_bounds, double C);
} // namespace recoloring
} // namespace cones