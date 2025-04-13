#pragma once

#include <assert.h>

#ifdef PARANOID
#define paranoid_assert(x) (assert(x))
#else
#define paranoid_assert(x) ((void)0)
#endif