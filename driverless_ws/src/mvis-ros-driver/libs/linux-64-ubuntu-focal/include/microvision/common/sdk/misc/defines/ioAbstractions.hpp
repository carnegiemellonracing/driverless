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
//! \brief
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
//! \brief
//!        streamposToInt64(const std::streampos& streampos)
//------------------------------------------------------------------------------
#include <fstream>
namespace microvision {
namespace common {
namespace sdk {
#if defined(_MSC_VER) & _MSC_VER < 1700 // begin: VS up to 2010
inline int64_t streamposToInt64(const std::streampos& streampos)
{
    // MSVC has a 64 bit file size, but accessible only through
    // the non-standard std::fpos::seekpos() method.
    // still visual studio 2010 does not calculate right size by itself
    const int64_t pos64 = streampos.seekpos() + std::streamoff(streampos) - _FPOSOFF(streampos.seekpos());

    return pos64;
}
#else // end: VS up to 2010; begin: VS from 2012 and linux
inline int64_t streamposToInt64(const std::streampos& streampos) { return std::streamoff(streampos); }
#endif // begin: VS from 2012 and linux
} // namespace sdk
} // namespace common
} // namespace microvision
