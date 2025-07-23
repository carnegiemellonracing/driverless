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
//! \date Mar 07, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace image {
//==============================================================================

//==============================================================================
//! \brief Enumeration of pixel formats.
//------------------------------------------------------------------------------
enum class PixelFormat : uint8_t
{
    Unknown = 0, //!< Pixel format not known.
    Mono8   = 1, //!< Monochrome, 8-bit per pixel
    Rgb8    = 2, //!< Red-Green-Blue, 8-bit per pixel
    Yuv420  = 3, //!< YUV 4:2:0, 8-bit per pixel
    Yuv422  = 4 //!< YUV 4:2:2, 8-bit per pixel
};

//==============================================================================
} // namespace image
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
