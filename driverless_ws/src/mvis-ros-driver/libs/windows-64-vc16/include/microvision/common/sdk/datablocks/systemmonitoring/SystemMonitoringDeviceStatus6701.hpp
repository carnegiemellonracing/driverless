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
//! \brief SystemMonitoringDeviceStatus
//!
//! The SystemMonitoringDeviceStatus data type provides information about connected devices:
//! Scanner, Cameras, GPS, IMU. If devices have lost connection to the perception ECU, this datatype is also used
//! to signal the loss of these devices.
//------------------------------------------------------------------------------
class SystemMonitoringDeviceStatus6701 final : public SpecializedDataContainer
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

    //! Type of the device.
    enum class DeviceType : uint8_t
    {
        Unknown = 0,
        Scanner = 1,
        Camera  = 2,
        Can     = 3,
        Wgs84   = 4,
        Gps     = 5,
        Imu     = 6
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.systemmonitoringdevicestatus6701"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    SystemMonitoringDeviceStatus6701();
    virtual ~SystemMonitoringDeviceStatus6701();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    /*!
	 * \brief Convert the object to a string.
	 * \return the object as a string.
	 */
    std::string toString() const;

    //! Set the device ID.
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }

    //! Set the device type.
    void setType(const DeviceType newType) { m_type = newType; }

    //! Set the device type information.
    void setTypeInformation(const uint8_t newTypeInformation) { m_typeInformation = newTypeInformation; }

    //! Set the state of the CAN message.
    void setState(const State newState) { m_state = newState; }

    //! Set the state information of the CAN message.
    void setStateInformation(const uint8_t newStateInformation) { m_stateInformation = newStateInformation; }

    //! Set the timestamp of the last missing message.
    void setUpdateMissing(const boost::posix_time::ptime newUpdateMissing) { m_updateMissing = newUpdateMissing; }

    //! Get the device ID.
    uint8_t getDeviceId() const { return m_deviceId; }

    //! Get the device type.
    DeviceType getType() const { return m_type; }

    //! Get the device type information.
    //
    // If the device type is "Scanner", the return value is
    //      0 for undefined
    //      1 for MVIS LUX
    //      2 for Scala
    //    100 for Third party LD-MRS
    //    101 for Third party LMS.
    //
    // If the device type is "DeviceTypeIMU", the return value is
    //      0 for undefined
    //      1 for Xsens
    //      2 for ThirdPartyOGpsImu
    //      3 for Genesys.
    //
    // Otherwise, the return value is 0 (undefined).
    uint8_t getTypeInformation() const { return m_typeInformation; }

    //! Get the state of the CAN message.
    State getState() const { return m_state; }

    //! Get the state information of the CAN message.
    uint8_t getStateInformation() const { return m_stateInformation; }

    //! Get the timestamp of the last missing message.
    boost::posix_time::ptime getUpdateMissing() const { return m_updateMissing; }

private:
    uint8_t m_deviceId{0};
    DeviceType m_type{DeviceType::Unknown};
    uint8_t m_typeInformation{0};
    State m_state{State::Init};
    uint8_t m_stateInformation{0};
    boost::posix_time::ptime m_updateMissing{};
}; // SystemMonitoringDeviceStatus6701

//==============================================================================

template<>
inline void readBE<SystemMonitoringDeviceStatus6701::DeviceType>(std::istream& is,
                                                                 SystemMonitoringDeviceStatus6701::DeviceType& value)
{
    uint8_t tmp;
    readBE(is, tmp);
    value = SystemMonitoringDeviceStatus6701::DeviceType(tmp);
}

//==============================================================================

template<>
inline void
writeBE<SystemMonitoringDeviceStatus6701::DeviceType>(std::ostream& os,
                                                      const SystemMonitoringDeviceStatus6701::DeviceType& value)
{
    const uint8_t tmp = static_cast<uint8_t>(value);
    writeBE(os, tmp);
}

//==============================================================================

template<>
inline void readBE<SystemMonitoringDeviceStatus6701::State>(std::istream& is,
                                                            SystemMonitoringDeviceStatus6701::State& value)
{
    uint8_t tmp;
    readBE(is, tmp);
    value = SystemMonitoringDeviceStatus6701::State(tmp);
}

//==============================================================================

template<>
inline void writeBE<SystemMonitoringDeviceStatus6701::State>(std::ostream& os,
                                                             const SystemMonitoringDeviceStatus6701::State& value)
{
    const uint8_t tmp = static_cast<uint8_t>(value);
    writeBE(os, tmp);
}

//==============================================================================
//==============================================================================
//==============================================================================

bool operator==(const SystemMonitoringDeviceStatus6701& lhs, const SystemMonitoringDeviceStatus6701& rhs);
bool operator!=(const SystemMonitoringDeviceStatus6701& lhs, const SystemMonitoringDeviceStatus6701& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
