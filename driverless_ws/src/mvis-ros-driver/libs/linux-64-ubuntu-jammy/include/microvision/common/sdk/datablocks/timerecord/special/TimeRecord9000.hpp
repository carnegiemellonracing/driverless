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
//! \date Mar 21, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/ClockType.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief TimeRecord
//!
//! This data type contains the local system time as well as a time signal that is provided by an external device,
//! e.g. a GPS receiver connected to the MVIS ECU. This data can be used for offline transformation of the time base
//! of the recorded data stream.
//------------------------------------------------------------------------------
class TimeRecord9000 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const uint8_t nbOfReserved = 4;

public:
    enum class FixMode : uint8_t
    {
        NotSeen = 0,
        None    = 1,
        Fix2D   = 2,
        Fix3D   = 3
    };

    using NtpTimeVector = std::vector<NtpTime>;
    using FixModeVector = std::vector<FixMode>;
    using ReservedArray = std::array<uint32_t, nbOfReserved>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.timerecord9000"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    TimeRecord9000();
    ~TimeRecord9000() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    ClockType getInternalClockType() const { return m_internalClockType; }
    ClockType getExternalClockType() const { return m_externalClockType; }
    const NtpTimeVector& getInternalClockTimes() const { return m_internalClockTimes; }
    const NtpTimeVector& getExternalClockTimes() const { return m_externalClockTimes; }
    const FixModeVector& getFixModes() const { return m_fixModes; }

    uint32_t getReserved(const uint8_t idx) const { return m_reserved.at(idx); }
    const ReservedArray& getReserved() const { return m_reserved; }

public:
    void setInternalClockType(const ClockType& clockType) { this->m_internalClockType = clockType; }
    void setExternalClockType(const ClockType& clockType) { this->m_externalClockType = clockType; }
    bool setTimesAndFixModes(const NtpTimeVector& externalTimes,
                             const NtpTimeVector& internalTimes,
                             const FixModeVector& fixModes);

protected:
    ClockType m_externalClockType{}; //!< Characteristics of the external clock used to record times.
    ClockType m_internalClockType{}; //!< Characteristics of the internal clock used to record times.
    NtpTimeVector m_externalClockTimes{}; //!< Vector of times as represented by external clock.
    NtpTimeVector m_internalClockTimes{}; //!< Vector of times as represented by internal clock.
    FixModeVector m_fixModes; //!< Vector representing fix modes used.

    ReservedArray m_reserved{{0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU}};
}; // TimeRecord9000

//==============================================================================

bool operator==(const TimeRecord9000& lhs, const TimeRecord9000& rhs);
bool operator!=(const TimeRecord9000& lhs, const TimeRecord9000& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
