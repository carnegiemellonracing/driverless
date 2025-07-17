#pragma once
#include "../cones.hpp"
#include "../../../lib/svm_lib.hpp" // this is really shite way to do this

namespace point_to_pixel {
namespace recoloring {
    TrackBounds recolor_cones(TrackBounds track_bounds, double C);
} // namespace recoloring
} // namespace point_to_pixel_to_pixel