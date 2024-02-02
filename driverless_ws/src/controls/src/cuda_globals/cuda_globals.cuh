#pragma once

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {

        extern __constant__ const float perturbation_std[action_dims * action_dims];

    }
}
