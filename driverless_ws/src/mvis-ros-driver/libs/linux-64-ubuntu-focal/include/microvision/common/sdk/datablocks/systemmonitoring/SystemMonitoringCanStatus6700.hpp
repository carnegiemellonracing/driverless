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
//! \brief SystemMonitoringCANStatus
//!
//! The SystemMonitoringCANStatus data type provides information about registered CAN frames.
//! The registration itself is done in the Worker.xml configuration file in SystemMonitoringWorker configuration block.
//! If registered CAN frames are not received by the perception ECU anymore, this datatype is also used to signal the
//! loss of these CAN frames.
//------------------------------------------------------------------------------
class SystemMonitoringCanStatus6700 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //! State of the device.
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
        bit00_notEnoughSignals = 0,
        bit03_errorWasPresent  = 3,
        bit04_dropRateHigh     = 4,
        bit05_jitterHigh       = 5,
        bit07_timeout          = 7
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.systemmonitoringcanstatus6700"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    SystemMonitoringCanStatus6700();
    virtual ~SystemMonitoringCanStatus6700();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Convert the object to a string.
    //!\return the object as a string.
    //----------------------------------------
    std::string toString() const;

    //! Set the device ID.
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }

    //! Set the CAN message identifier.
    void setMessageId(const uint32_t newMessageId) { m_messageId = newMessageId; }

    //! Set the state of the CAN message.
    void setState(const State newState) { m_state = newState; }

    //! Set the state information of the CAN message.
    void setStateInformation(const uint8_t newStateInformation) { m_stateInformation = newStateInformation; }

    //! Set the timestamp of the last missing message.
    void setUpdateMissing(const boost::posix_time::ptime newUpdateMissing) { m_updateMissing = newUpdateMissing; }

    //! Get the device ID.
    uint8_t getDeviceId() const { return m_deviceId; }

    //! Get the CAN message identifier.
    uint32_t getMessageId() const { return m_messageId; }

    //! Get the state of the CAN message.
    State getState() const { return m_state; }

    //! Get the state information of the CAN message.
    uint8_t getStateInformation() const { return m_stateInformation; }

    //! Get the timestamp of the last missing message.
    boost::posix_time::ptime getUpdateMissing() const { return m_updateMissing; }

private:
    uint8_t m_deviceId{0};
    uint32_t m_messageId{0};
    State m_state{State::Init};
    uint8_t m_stateInformation{0};
    boost::posix_time::ptime m_updateMissing{};
}; // SystemMonitoringCanStatus6700

//==============================================================================

template<>
inline void readBE<SystemMonitoringCanStatus6700::State>(std::istream& is, SystemMonitoringCanStatus6700::State& value)
{
    uint8_t tmp;
    readBE(is, tmp);
    value = SystemMonitoringCanStatus6700::State(tmp);
}

//==============================================================================

template<>
inline void writeBE<SystemMonitoringCanStatus6700::State>(std::ostream& os,
                                                          const SystemMonitoringCanStatus6700::State& value)
{
    const uint8_t tmp = static_cast<uint8_t>(value);
    writeBE(os, tmp);
}

//==============================================================================

bool operator==(const SystemMonitoringCanStatus6700& lhs, const SystemMonitoringCanStatus6700& rhs);
bool operator!=(const SystemMonitoringCanStatus6700& lhs, const SystemMonitoringCanStatus6700& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
