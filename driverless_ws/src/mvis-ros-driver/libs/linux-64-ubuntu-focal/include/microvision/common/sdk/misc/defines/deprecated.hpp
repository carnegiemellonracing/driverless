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
//! \brief Defines for deprecating code.
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
//! \brief Defines for deprecating code.
//!        ALLOW_WARNING_DEPRECATED
//!        MICROVISION_SDK_DEPRECATED
//------------------------------------------------------------------------------

// Check for C++14 or newer
#if __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900)
// Use standard C++14 [[deprecated]] attribute
#    define MICROVISION_SDK_DEPRECATED [[deprecated]]

// Define warning suppression based on compiler
#    ifdef __GNUC__
#        define ALLOW_WARNING_DEPRECATED _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#    elif defined(__clang__)
#        define ALLOW_WARNING_DEPRECATED _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#    elif defined(_MSC_VER)
#        define ALLOW_WARNING_DEPRECATED __pragma(warning(disable : 4996))
#    else
#        define ALLOW_WARNING_DEPRECATED
#    endif

#else
// Fall back to compiler-specific approaches for older compilers
#    ifdef __GNUC__
#        define MICROVISION_SDK_DEPRECATED __attribute__((deprecated))
#        define ALLOW_WARNING_DEPRECATED _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#    elif defined(__clang__)
#        define MICROVISION_SDK_DEPRECATED __attribute__((deprecated))
#        define ALLOW_WARNING_DEPRECATED _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#    elif defined(_MSC_VER)
#        define MICROVISION_SDK_DEPRECATED __declspec(deprecated)
#        define ALLOW_WARNING_DEPRECATED __pragma(warning(disable : 4995)) __pragma(warning(disable : 4996))
// 4995: name was marked as #pragma deprecated
// 4996: __declspec(deprecated)
#    else
#        define MICROVISION_SDK_DEPRECATED
#        define ALLOW_WARNING_DEPRECATED
#    endif
#endif
