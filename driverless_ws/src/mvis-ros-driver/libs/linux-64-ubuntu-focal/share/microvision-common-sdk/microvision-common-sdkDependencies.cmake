if (DEFINED MICROVISION_microvision-common-sdk_INCLUDE_GUARD)
	return()
endif()
set(MICROVISION_microvision-common-sdk_INCLUDE_GUARD 1)

include(CMakeFindDependencyMacro)
find_dependency(microvision-common-logging REQUIRED)
find_dependency(Boost REQUIRED COMPONENTS thread system program_options date_time filesystem unit_test_framework)
find_dependency(JPEG REQUIRED)
