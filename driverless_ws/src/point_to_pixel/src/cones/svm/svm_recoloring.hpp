#pragma once

#include "cones.hpp"
#include "svm.hpp"

namespace cones {
    namespace recoloring {
        namespace first_svm {
            /**
             * @brief Recolors cones by training SVM on raw cones, then running
             * SVM on each cone to rectify incorrectly colored cones.
             * 
             * @param Trackbounds struct containing vectors of yellow and blue cones
             * @return struct containing vectors of corrected yellow and blue cones
             */
            cones::TrackBounds recolor_cones(cones::TrackBounds track_bounds);
        }
    }
}