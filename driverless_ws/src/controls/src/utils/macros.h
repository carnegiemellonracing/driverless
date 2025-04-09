#pragma once

#include <model/slipless/model.cuh>
#include <model/sysid/model.cuh>

// ^ I think this is a macro because we didn't want to use a function object
#define ONLINE_DYNAMICS_FUNC controls::model::sysid::dynamics