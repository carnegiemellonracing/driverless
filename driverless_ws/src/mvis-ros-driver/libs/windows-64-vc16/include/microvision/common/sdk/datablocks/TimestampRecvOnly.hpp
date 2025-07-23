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

#include <microvision/common/sdk/datablocks/ClockType.hpp>
#include <microvision/common/sdk/datablocks/TimestampInterface.hpp>

#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Math.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class TimestampRecvOnly final : public TimestampInterface
{
public:
    TimestampRecvOnly();
    TimestampRecvOnly(const NtpTime receivedTimeEcu, const NtpTime receivedTime, const ClockType clockType);
    virtual ~TimestampRecvOnly();

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public:
    NtpTime getReceivedTime() const override { return m_receivedTime; }
    NtpTime getReceivedTimeEcu() const override { return m_receivedTimeEcu; }
    ClockType getClockType() const override { return m_clockType; }

    NtpTime getMeasurementTime() const override
    {
        LOGTRACE(logger, "Measurement time not available, returning the closest estimate, ReceivedTime");
        return m_receivedTime;
    }

    NtpTime getMeasurementTimeEcu() const override
    {
        LOGTRACE(logger, "Measurement time not available, returning the closest estimate, ReceivedTimeECU");
        return m_receivedTimeEcu;
    }

    NtpTime getRawDeviceTime() const override
    {
        LOGTRACE(logger, "Not available");
        return false;
    }

    bool hasMeasurementTimeEcu() const override { return false; }
    bool hasMeasurementTime() const override { return false; }

public:
    void setReceivedTime(const NtpTime& receivedTime) { this->m_receivedTime = receivedTime; }
    void setReceivedTimeEcu(const NtpTime& receivedTimeEcu) { this->m_receivedTimeEcu = receivedTimeEcu; }
    void setClockType(const ClockType& clockType) { this->m_clockType = clockType; }

private:
    static constexpr const char* loggerId = "microvision::common::sdk::TimestampRecvOnly";
    static microvision::common::logging::LoggerSPtr logger;

private:
    NtpTime m_receivedTimeEcu;
    NtpTime m_receivedTime;
    ClockType m_clockType;
}; // TimestampRecvOnly

//==============================================================================

bool operator==(const TimestampRecvOnly& ts1, const TimestampRecvOnly& ts2);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
