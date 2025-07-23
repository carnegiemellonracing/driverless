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
//! \date Apr 29, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/timerecord/special/TimeRecord9000.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This data type contains the local system time as well as a time signal that is provided by an external device
//!
//! Special data types:
//! \ref microvision::common::sdk::TimeRecord9000
//------------------------------------------------------------------------------
class TimeRecord final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const TimeRecord&, const TimeRecord&);

public:
    using FixMode       = TimeRecord9000::FixMode;
    using NtpTimeVector = TimeRecord9000::NtpTimeVector;
    using FixModeVector = TimeRecord9000::FixModeVector;
    using ReservedArray = TimeRecord9000::ReservedArray;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.timerecord"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    TimeRecord();
    ~TimeRecord() override = default;

    TimeRecord(const TimeRecord& rhs) = default;
    TimeRecord& operator=(const TimeRecord& rhs) = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    ClockType getInternalClockType() const { return m_delegate.getInternalClockType(); }
    ClockType getExternalClockType() const { return m_delegate.getExternalClockType(); }
    const NtpTimeVector& getInternalClockTimes() const { return m_delegate.getInternalClockTimes(); }
    const NtpTimeVector& getExternalClockTimes() const { return m_delegate.getExternalClockTimes(); }
    const FixModeVector& getFixModes() const { return m_delegate.getFixModes(); }

    uint32_t getReserved(const uint8_t idx) const { return m_delegate.getReserved(idx); }
    const ReservedArray& getReserved() const { return m_delegate.getReserved(); }

public: // setter
    void setInternalClockType(const ClockType& clockType) { m_delegate.setInternalClockType(clockType); }
    void setExternalClockType(const ClockType& clockType) { m_delegate.setExternalClockType(clockType); }
    bool setTimesAndFixModes(const NtpTimeVector& externalTimes,
                             const NtpTimeVector& internalTimes,
                             const FixModeVector& fixModes)
    {
        return m_delegate.setTimesAndFixModes(externalTimes, internalTimes, fixModes);
    }

protected:
    TimeRecord9000 m_delegate; // only possible specialization currently
}; // TimeRecord

//==============================================================================

inline bool operator==(const TimeRecord& lhs, const TimeRecord& rhs) { return lhs.m_delegate == rhs.m_delegate; }
inline bool operator!=(const TimeRecord& lhs, const TimeRecord& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
