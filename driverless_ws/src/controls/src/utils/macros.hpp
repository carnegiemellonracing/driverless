#pragma once

#include <model/sysid/sysid_model_host.h>
#include <model/sysid/sysid_model_sim.h>

#include <model/slipless/slipless_model_host.h>


// ^ I think this is a macro because we didn't want to use a function object
#ifdef USESYSID
#define HOST_DYNAMICS_FUNC controls::model_host::sysid::dynamics
#define SIM_DYNAMICS_FUNC controls::model_sim::sysid::dynamics
#else
#define HOST_DYNAMICS_FUNC controls::model_host::slipless::dynamics
#define SIM_DYNAMICS_FUNC controls::model_host::slipless::dynamics

#endif
