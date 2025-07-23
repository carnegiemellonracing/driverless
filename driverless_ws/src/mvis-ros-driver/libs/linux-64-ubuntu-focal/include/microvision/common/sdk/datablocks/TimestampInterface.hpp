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
//! \date Mar 28, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class TimestampInterface
{
public:
    TimestampInterface(){};
    virtual ~TimestampInterface(){};

public:
    virtual NtpTime getReceivedTime() const       = 0;
    virtual NtpTime getMeasurementTime() const    = 0;
    virtual NtpTime getReceivedTimeEcu() const    = 0;
    virtual NtpTime getMeasurementTimeEcu() const = 0;
    virtual ClockType getClockType() const        = 0;
    virtual NtpTime getRawDeviceTime() const      = 0;
    virtual bool hasMeasurementTimeEcu() const    = 0;
    virtual bool hasMeasurementTime() const       = 0;
};

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
