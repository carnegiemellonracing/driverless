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
//! \date Jan 30, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6301.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class DeviceStatus6301SerializedSize6301 final
{
public:
    static std::streamsize getSerializedSize(const DeviceStatus6301& ds);
    static std::streamsize getSerializedSize(const SerialNumberIn6301& sn);
    static std::streamsize getSerializedSize(const Version448In6301& version);
}; // DeviceStatus6301SerializedSize6301

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
