#pragma once

#include "controller.hpp"

namespace controls {
    /**
     * Dynamical model of car. Calculates time-derivative of the vehicle state
     * given a state and control action. This forms the RHS of the differential
     * equation describing vehicle behavior.
     *
     * @param vehicleState Current vehicle state
     * @param controlAction Control action
     * @param vehicleStateDotData Buffer to write state derivative
     */
    void model_dynamics(const VehicleState& vehicleState,
                        const ControlAction& controlAction,
                        double vehicleStateDotData[VEHICLE_STATE_DIMS]);
}
