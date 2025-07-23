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

#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6303.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class DeviceStatus6303SerializedSize6303 final
{
public:
    static std::streamsize getSerializedSize(const DeviceStatus6303& ds);
    static std::streamsize getSerializedSize(const SerialNumberIn6303& sn);
    static std::streamsize getSerializedSize(const Version448In6303& version);
}; // DeviceStatus6303SerializedSize6303

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
