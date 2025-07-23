if (DEFINED MICROVISION_microvision-common-sdk-test-support-static_INCLUDE_GUARD)
	return()
endif()
set(MICROVISION_microvision-common-sdk-test-support-static_INCLUDE_GUARD 1)

include(CMakeFindDependencyMacro)
find_dependency(microvision-common-sdk-static)
