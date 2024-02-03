#include "../types.hpp"
#include "../cuda_utils.cuh"

namespace controls {
    namespace mppi {
        struct ActionWeightTuple {
            Action action;
            float weight;
        };
    }
}
