#pragma once

#include <interfaces/msg/spline_list.hpp>
#include <array>


namespace controls {
    constexpr size_t action_dims = 2;
    constexpr size_t state_dims = 13;

    using Action = std::array<double, action_dims>;
    using State = std::array<double, state_dims>;

    using SplineMsg = interfaces::msg::SplineList;
    using SlamMsg = struct {};
    using GpsMsg = struct {};

    enum class Device {
        Cpu,
        Cuda
    };

    class Controller {
    public:
        virtual Action generate_action(const State &current_state) =0;

        virtual ~Controller() =0;
    };
}
