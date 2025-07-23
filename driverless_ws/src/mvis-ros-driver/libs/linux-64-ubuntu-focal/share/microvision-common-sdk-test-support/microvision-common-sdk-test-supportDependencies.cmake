if (DEFINED MICROVISION_microvision-common-sdk-test-support_INCLUDE_GUARD)
	return()
endif()
set(MICROVISION_microvision-common-sdk-test-support_INCLUDE_GUARD 1)

include(CMakeFindDependencyMacro)
find_dependency(microvision-common-sdk)
