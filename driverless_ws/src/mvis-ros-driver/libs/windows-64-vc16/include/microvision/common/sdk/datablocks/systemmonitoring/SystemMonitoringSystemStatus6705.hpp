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
//! \date Jan 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief SystemMonitoringSystemStatus
//!
//! The SystemsMonitoringSystemStatus data type provides information about the current state of the perception ECU
//! combined with the information about the entire fusion system.
//------------------------------------------------------------------------------
class SystemMonitoringSystemStatus6705 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //! State of the system.
    enum class State : uint8_t
    {
        Init    = 0,
        OK      = 1,
        Warning = 2,
        Error   = 3
    };

    //! State information bits
    enum class StateInformationBits : uint8_t
    {
        bit00_highLatency      = 0,
        bit03_memoryAlmostFull = 3,
        bit05_timeout          = 5,
        bit06_memoryFull       = 6
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.systemmonitoringsystemstatus6705"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    SystemMonitoringSystemStatus6705();
    virtual ~SystemMonitoringSystemStatus6705();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    /*!
	 * \brief Convert the object to a string.
	 * \return the object as a string.
	 */
    std::string toString() const;

public: // getter
    //! Get the time stamp of the last update.
    const boost::posix_time::ptime getLastUpdateTimestamp() const { return m_lastUpdateTimestamp; }

    //! Get the system state.
    State getState() const { return m_state; }

    //! Get the system state information.
    uint8_t getStateInformation() const { return m_stateInformation; }

    //! Get the CPU usage.
    uint8_t getCurrentCpuUsage() const { return m_currentCpuUsage; }

    //! Get the RAM usage.
    uint8_t getCurrentRamUsage() const { return m_currentRamUsage; }

    //! Get the HDD usage.
    uint8_t getCurrentHddUsage() const { return m_currentHddUsage; }

    //! Get the HDD warning level.
    uint8_t getHddWarningLevel() const { return m_hddWarningLevelPercent; }

    //! Get the HDD error level.
    uint8_t getHddErrorLevel() const { return m_hddErrorLevelPercent; }

public: // setter
    //! Set the time stamp of the last update.
    void setLastUpdateTimestamp(const boost::posix_time::ptime newLastUpdateTimestamp)
    {
        m_lastUpdateTimestamp = newLastUpdateTimestamp;
    }

    //! Set the system state.
    void setState(const State newState) { m_state = newState; }

    //! Set the system state information.
    void setStateInformation(const uint8_t newStateInformation) { m_stateInformation = newStateInformation; }

    //! Set the CPU usage.
    void setCurrentCpuUsage(const uint8_t newCpuUsage) { m_currentCpuUsage = newCpuUsage; }

    //! Set the RAM usage.
    void setCurrentRamUsage(const uint8_t newRamUsage) { m_currentRamUsage = newRamUsage; }

    //! Set the HDD usage.
    void setCurrentHddUsage(const uint8_t newHddUsage) { m_currentHddUsage = newHddUsage; }

    //! Set the HDD warning level.
    void setHddWarningLevel(const uint8_t newHddWarningLevel) { m_hddWarningLevelPercent = newHddWarningLevel; }

    //! Set the HDD error level.
    void setHddErrorLevel(const uint8_t newHddErrorLevel) { m_hddErrorLevelPercent = newHddErrorLevel; }

private:
    boost::posix_time::ptime m_lastUpdateTimestamp{};
    State m_state{};
    uint8_t m_stateInformation{static_cast<uint8_t>(State::Init)};
    uint8_t m_currentCpuUsage{0};
    uint8_t m_currentRamUsage{0};
    uint8_t m_currentHddUsage{0};
    uint8_t m_hddWarningLevelPercent{0};
    uint8_t m_hddErrorLevelPercent{0};
}; // SystemMonitoringSystemStatus6705

//==============================================================================

template<>
inline void readBE<SystemMonitoringSystemStatus6705::State>(std::istream& is,
                                                            SystemMonitoringSystemStatus6705::State& value)
{
    uint8_t tmp;
    readBE(is, tmp);
    value = SystemMonitoringSystemStatus6705::State(tmp);
}

//==============================================================================

template<>
inline void writeBE<SystemMonitoringSystemStatus6705::State>(std::ostream& os,
                                                             const SystemMonitoringSystemStatus6705::State& value)
{
    const uint8_t tmp = static_cast<uint8_t>(value);
    writeBE(os, tmp);
}
//==============================================================================

//==============================================================================

bool operator==(const SystemMonitoringSystemStatus6705& lhs, const SystemMonitoringSystemStatus6705& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
