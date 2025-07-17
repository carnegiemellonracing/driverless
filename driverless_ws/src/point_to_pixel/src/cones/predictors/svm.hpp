#pragma once
#include "../cones.hpp"
#include "../../../lib/svm_lib.hpp" // this is really shite way to do this

namespace point_to_pixel {
namespace recoloring {
    /**
     * @brief applies support vector machines to correct for misclassified cones. The algorithm takes in two sets of 
     * cones, calculates the best seperation boundary between them, and then reclassifies the cones
     * 
     * @param track_bounds set of yellow and set of blue Cone objects
     * @param C C param for svm controls trade-off between correct classification and maximum margin seperation boundary
     * @returns TrackBounds object that represents a newly reclassified left and right set of cones
     */
    TrackBounds recolor_cones(TrackBounds track_bounds, double C);
} // namespace recoloring
} // namespace point_to_pixel_to_pixel