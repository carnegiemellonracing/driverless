#pragma once

#include <model/slipless/slipless_model.cuh>
#include <model/sysid/sysid_model.cuh>
#include <model/sysid/sysid_model_host.h>
#include <model/slipless/slipless_model_host.h>

// ^ I think this is a macro because we didn't want to use a function object
#define ONLINE_DYNAMICS_FUNC controls::model::sysid::dynamics
#define HOST_DYNAMICS_FUNC controls::model_host::sysid::dynamics
#define CENTRIPEDAL_ACCEL_FUNC controls::model::sysid::centripedal_accel