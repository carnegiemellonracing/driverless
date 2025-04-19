#pragma once

#include <model/sysid/sysid_model_host.h>
#include <model/slipless/slipless_model_host.h>
#include <model/steering/controller_steering_model_host.h>


// ^ I think this is a macro because we didn't want to use a function object
#ifdef USESYSID
#define HOST_DYNAMICS_FUNC controls::model_host::sysid::dynamics
//#define HOST_DYNAMICS_FUNC controls::model_host::steering::dynamics
#else
#define HOST_DYNAMICS_FUNC controls::model_host::slipless::dynamics
#endif
