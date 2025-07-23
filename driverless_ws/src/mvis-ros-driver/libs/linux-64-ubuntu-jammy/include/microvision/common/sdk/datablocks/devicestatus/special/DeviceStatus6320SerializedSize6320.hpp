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
//! \date Jun 24, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6320.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class DeviceStatus6320SerializedSize6320 final
{
public:
    static std::streamsize getSerializedSize(const DeviceStatus6320& ds);
    static std::streamsize getSerializedSizeOfErrorIn6320();
}; // DeviceStatus6320SerializedSize6320

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
