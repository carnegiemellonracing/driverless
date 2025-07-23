//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 28, 2021
//------------------------------------------------------------------------------

//==============================================================================
//! \brief Defines for enable or disable warnings.
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
//! \brief Defines for enable or disable warnings.
//!        ALLOW_WARNINGS_BEGIN
//!        ALLOW_WARNINGS_END
//------------------------------------------------------------------------------
#ifdef __GNUC__
#    ifndef __GNUC_PREREQ
#        /* Mingw has gcc but not this particular version check (huh?!?) */
#        define __GNUC_PREREQ(a, b) 1
#    endif
#    if __GNUC_PREREQ(3, 0)
#        define ALLOW_WARNINGS_BEGIN _Pragma("GCC diagnostic push")
#        define ALLOW_WARNINGS_END _Pragma("GCC diagnostic pop")
#    endif

#elif defined(__clang__)
#    define ALLOW_WARNINGS_BEGIN _Pragma("GCC diagnostic push")
#    define ALLOW_WARNINGS_END _Pragma("GCC diagnostic pop")

#else /* Non-gcc compiler */
#    ifdef _MSC_VER
#        define ALLOW_WARNINGS_BEGIN __pragma(warning(push))
#        define ALLOW_WARNINGS_END __pragma(warning(pop))
#    endif
#endif

//==============================================================================
//! \brief Defines for enable or disable the switch-enum warning.
//!        SWITCH_ENUM_WARNING_ENABLE
//!        SWITCH_ENUM_WARNING_DISABLE
//------------------------------------------------------------------------------
#ifdef __GNUC__
#    define SWITCH_ENUM_WARNING_ENABLE _Pragma("GCC diagnostic error \"-Wswitch-enum\"")
#    define SWITCH_ENUM_WARNING_DISABLE _Pragma("GCC diagnostic ignored \"-Wswitch-enum\"")

#elif defined(__clang__)
#    define SWITCH_ENUM_WARNING_ENABLE _Pragma("GCC diagnostic error \"-Wswitch-enum\"")
#    define SWITCH_ENUM_WARNING_DISABLE _Pragma("GCC diagnostic ignored \"-Wswitch-enum\"")

#else /* Non-gcc compiler */
#    ifdef _MSC_VER
#        define SWITCH_ENUM_WARNING_ENABLE __pragma(warning(error : 4061)) __pragma(warning(error : 4062))
#        define SWITCH_ENUM_WARNING_DISABLE __pragma(warning(disable : 4061)) __pragma(warning(disable : 4062))
#        // 4061: enumerator 'identifier' in switch of enum 'enumeration' is not explicitly handled by a case label
#        // 4062: enumerator 'identifier' in switch of enum 'enumeration' is not handled
#    else
#        define SWITCH_ENUM_WARNING_ENABLE
#        define SWITCH_ENUM_WARNING_DISABLE
#    endif
#endif