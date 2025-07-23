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
//! \brief Defines for windows compatibility.
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
//! \brief
//!        __BIG_ENDIAN
//!        __LITTLE_ENDIAN
//------------------------------------------------------------------------------
#ifdef _WIN32
#    ifndef _USE_MATH_DEFINES
#        define _USE_MATH_DEFINES
#    endif // _USE_MATH_DEFINES

#    ifndef NOMINMAX
#        define NOMINMAX
#    endif // NOMINMAX

#    define __func__ __FUNCTION__

#    ifndef __BIG_ENDIAN
#        define __BIG_ENDIAN 1234
#    endif // __BIG_ENDIAN

#    ifndef __LITTLE_ENDIAN
#        define __LITTLE_ENDIAN 3412
#    endif // __LITTLE_ENDIAN

#endif // _WIN32

//==============================================================================
//! \brief
//!        __declspec (nothrow)
//!        __declspec(deprecated)
//------------------------------------------------------------------------------
#ifdef _WIN32
#    if _MSC_VER <= 1910
#        pragma warning(disable : 4290)
#        pragma warning(disable : 4996)
#    endif
#endif // _WIN32

//==============================================================================
//! \brief
//!        MICROVISION_SDKLIBDLL_EXPORTS
//------------------------------------------------------------------------------
#ifdef _WIN32
#    ifdef MICROVISION_SDKLIBDLL_EXPORTS
#        define MICROVISION_SDKLIBDLL_API __declspec(dllexport)
#    else
#        define MICROVISION_SDKLIBDLL_API __declspec(dllimport)
#    endif
#endif // WIN32
