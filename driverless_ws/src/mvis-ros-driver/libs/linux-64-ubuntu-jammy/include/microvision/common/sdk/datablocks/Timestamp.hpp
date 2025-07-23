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

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class Timestamp final : public microvision::common::sdk::TimestampInterface
{
public:
    Timestamp();
    Timestamp(const NtpTime measurementTimeEcu,
              const NtpTime receivedTimeEcu,
              const NtpTime rawDeviceTime,
              const NtpTime measurementTime,
              const NtpTime receivedTime,
              const ClockType clockType);
    Timestamp(const NtpTime measurementTimeEcu, const NtpTime receivedTimeEcu);
    virtual ~Timestamp();

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    virtual NtpTime getReceivedTime() const override { return m_receivedTime; }
    virtual NtpTime getReceivedTimeEcu() const override { return m_receivedTimeEcu; }
    virtual ClockType getClockType() const override { return m_clockType; }
    virtual NtpTime getMeasurementTime() const override { return m_measurementTime; }
    virtual NtpTime getMeasurementTimeEcu() const override { return m_measurementTimeEcu; }
    virtual NtpTime getRawDeviceTime() const override { return m_rawDeviceTime; }
    virtual bool hasMeasurementTimeEcu() const override { return true; }
    virtual bool hasMeasurementTime() const override { return true; }

public:
    void setReceivedTime(const NtpTime& receivedTime) { this->m_receivedTime = receivedTime; }
    void setReceivedTimeECU(const NtpTime& receivedTimeEcu) { this->m_receivedTimeEcu = receivedTimeEcu; }
    void setRawDeviceTime(const NtpTime& rawDeviceTime) { this->m_rawDeviceTime = rawDeviceTime; }
    void setMeasurementTime(const NtpTime& measurementTime) { this->m_measurementTime = measurementTime; }
    void setMeasurementTimeEcu(const NtpTime& measurementTimeEcu) { this->m_measurementTimeEcu = measurementTimeEcu; }
    void setClockType(const ClockType& clockType) { this->m_clockType = clockType; }

private:
    NtpTime m_measurementTimeEcu;
    NtpTime m_receivedTimeEcu;
    NtpTime m_rawDeviceTime;
    NtpTime m_measurementTime;
    NtpTime m_receivedTime;
    ClockType m_clockType;
}; // Timestamp

//==============================================================================

bool operator==(const Timestamp& ts1, const Timestamp& ts2);
bool operator!=(const Timestamp& ts1, const Timestamp& ts2);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
