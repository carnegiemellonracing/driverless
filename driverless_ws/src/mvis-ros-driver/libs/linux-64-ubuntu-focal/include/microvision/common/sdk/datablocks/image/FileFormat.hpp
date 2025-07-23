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
//! \brief Enumeration of file formats.
//------------------------------------------------------------------------------
enum class FileFormat : uint8_t
{
    Unknown = 0, //!< File format not known.
    Raw     = 1, //!< Raw image buffer (no header, just plain image data).
    Jpeg    = 2, //!< Joint Photographic Experts Group file format.
    Mjpeg   = 3, //!< Motion JPEG format (single image of an MJPEG stream).
    Png     = 4 //!< Portable Network Graphics file format.
    //        Bitmap  = 5 //!< Windows Bitmap file format (including bitmap file header and DIB header).
};

//==============================================================================
} // namespace image
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
