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

#include <microvision/common/sdk/datablocks/devicestatus/DeviceStatus.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class DeviceStatusSerializedSize6301 final
{
public:
    static std::streamsize getSerializedSize(const DeviceStatus& ds);
    static std::streamsize getSerializedSize(const SerialNumber& sn);
    static std::streamsize getSerializedSize(const Version448& version);
}; // DeviceStatus6301SerializedSize6301

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
