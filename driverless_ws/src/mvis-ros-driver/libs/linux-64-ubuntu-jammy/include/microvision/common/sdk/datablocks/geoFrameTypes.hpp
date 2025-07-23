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
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

using TileIndex    = std::pair<int32_t, int32_t>;
using TileSizeType = uint32_t;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace std {
template<>
struct hash<microvision::common::sdk::TileIndex>
{
    size_t operator()(const microvision::common::sdk::TileIndex& t) const { return std::hash<int32_t>()(t.first); }
};
} // namespace std
//==============================================================================
