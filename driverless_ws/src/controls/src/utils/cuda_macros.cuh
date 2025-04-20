#pragma once

#include <utils/macros.hpp>
#include <model/slipless/slipless_model.cuh>
#include <model/sysid/sysid_model.cuh>
#include <model/steering/steering_model.cuh>

// ^ I think this is a macro because we didn't want to use a function object

#ifdef STEERING_MODEL
    #define ONLINE_DYNAMICS_FUNC controls::model::steering::dynamics
    #define CENTRIPEDAL_ACCEL_FUNC controls::model::sysid::centripedal_accel
#else
    #ifdef USESYSID
    #define ONLINE_DYNAMICS_FUNC controls::model::sysid::dynamics
    #define CENTRIPEDAL_ACCEL_FUNC controls::model::sysid::centripedal_accel
    #else
    #define ONLINE_DYNAMICS_FUNC controls::model::slipless::dynamics
    #define CENTRIPEDAL_ACCEL_FUNC controls::model::slipless::centripedal_accel
    #endif
#endif