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
//! \brief General defines.
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
//! \brief Macro to mark the symbol as import/export.
//------------------------------------------------------------------------------
#ifdef _WIN32
#    if defined MICROVISION_SDKLIBDLL_SHARED // shared usage
#        if defined MICROVISION_SDKLIBDLL_EXPORTS // building --> exporting
#            define MICROVISION_SDK_API __declspec(dllexport)
#        else // using --> import
#            define MICROVISION_SDK_API __declspec(dllimport)
#        endif
#    else // static case for windows
#        define MICROVISION_SDK_API
#    endif

#else // not windows
#    define MICROVISION_SDK_API
#endif // _WIN32

//==============================================================================
//! \brief An embedded auto include for the stdInt data type.
//!        include <stdint.h>
//------------------------------------------------------------------------------
#if _MSC_VER == 1500
#    ifdef WCHAR_MIN
#        undef WCHAR_MIN
#        undef INT8_C
#        undef UINT8_C
#        undef INT16_C
#        undef UINT16_C
#        undef INT32_C
#        undef UINT32_C
#        undef INT64_C
#        undef UINT64_C
#        undef INTMAX_C
#        undef UINTMAX_C
#    endif // WCHAR_MIN
#    include <stdintForVS2008.h> // using a copy of VS2010 stdint.h here
#else
#    include <stdint.h>
#endif

//==============================================================================
//! \brief Used std version of sdk to enable/disable features.
//------------------------------------------------------------------------------
#ifdef _MSC_FULL_VER
#    if _MSC_FULL_VER > 191426433 // Visual Studio 2017 Version 15.8
#        if __cplusplus >= 201703
#            define SDK_CPPSTD_VERSION 201703L // c++17
#        else // no need for the exact value here VS 2017 supports C++14 or C++17 only
#            define SDK_CPPSTD_VERSION 201402L // c++14
#        endif
#    elif _MSC_FULL_VER >= 190024210 // Visual Studio 2015 Update 3 Version 14.0
#        define SDK_CPPSTD_VERSION 201402L // c++14
#    else
#        define SDK_CPPSTD_VERSION 201103L // c++11
#    endif
#else // not msvc compiler
#    define SDK_CPPSTD_VERSION __cplusplus
#endif