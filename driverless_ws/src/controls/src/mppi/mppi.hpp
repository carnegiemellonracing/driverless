#pragma once


namespace controls {
    namespace mppi {
        class MppiController : public Controller { };
    }
}


#include "cpu_mppi.hpp"

#ifndef CONTROLS_NO_CUDA
#include "cuda_mppi.cuh"
#endif