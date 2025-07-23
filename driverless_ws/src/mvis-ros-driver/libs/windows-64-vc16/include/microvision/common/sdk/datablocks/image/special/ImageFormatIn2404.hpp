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
//! \date Nov 6th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <stdint.h>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace image {
//==============================================================================

//==============================================================================
//! \brief Enumeration of image formats.
//------------------------------------------------------------------------------
enum class ImageFormatIn2404 : uint16_t
{
    Jpeg   = 0, // JPEG compressed image
    Mjpeg  = 1, // Motion JPEG
    Gray8  = 2, // Monochrome, 8 bits-per-pixel grayscale channel, allowing 256 shades of gray.
    Yuv420 = 3, // Full luma, 2:1 horizontal and vertical chroma channel format image
    Yuv422 = 4, // Full luma, 2:1 horizontal chroma channel format image
    Png    = 5 // PNG compressed image.
};

//==============================================================================
//! \brief Converts the image format to a human readable string.
//!
//! \param[in] format  Image2404 format id to be converted into a string.
//! \return Name of the image format.
//------------------------------------------------------------------------------
std::string formatToString(const ImageFormatIn2404 format);

//==============================================================================
} // namespace image
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
