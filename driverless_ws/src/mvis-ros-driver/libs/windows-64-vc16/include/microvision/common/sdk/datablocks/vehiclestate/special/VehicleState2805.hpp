//==============================================================================
//! \file
//!
//!vehicle state data type 0x2805 send by LUX/Scala
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 2, 2013
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
//! \brief State Data in the format of LUX3-Firmware, or Scala Firmware
//!
//! The vehicle state is calculated by MVIS LUX from received CAN-Data.
//! CAN data parsers need to be configured in order to receive valid vehicle state information.
//!
//! All angles, position and distances are given in the ISO 8855 / DIN 70000 scanner coordinate system.
//!
//! General data type: \ref microvision::common::sdk::VehicleState
//------------------------------------------------------------------------------
class VehicleState2805 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    enum class ErrorFlags : uint16_t
    {
        ParamAxleDistNotSet = 0x0001U, //!< Axle dist parameter is not set, i.e. is set to zero.
        LastSwaNotUpToDate  = 0x0100U, //!< There is no latest (current) measurement of steering wheel angle (SWA).

        //! There is no latest (current) measurement of front wheel angle, or could not be calculated by SWA
        LastFwaNotUpToDate = 0x0200U,

        BufferEmpty = 0x0800U //!< No CAN data received
    }; // ErrorFlags

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.vehiclestate2805"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    VehicleState2805();
    ~VehicleState2805() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    NtpTime getTimestamp() const { return this->m_timestamp; }
    uint16_t getScanNumber() const { return this->m_scanNumber; }
    uint16_t getErrorFlags() const { return this->m_errorFlags; }
    int16_t getLongitudinalVelocity() const { return this->m_longitudinalVelocity; }
    int16_t getSteeringWheelAngle() const { return this->m_steeringWheelAngle; }
    int16_t getWheelAngle() const { return this->m_wheelAngle; }
    int32_t getXPosition() const { return this->m_xPos; }
    int32_t getYPosition() const { return this->m_yPos; }
    int16_t getCourseAngle() const { return this->m_courseAngle; }
    uint16_t getTimeDiff() const { return this->m_timeDiff; }
    int16_t getXDiff() const { return this->m_xDiff; }
    int16_t getYDiff() const { return this->m_yDiff; }
    int16_t getYaw() const { return this->m_yaw; }
    int16_t getCurrentYawRate() const { return this->m_currentYawRate; }
    int16_t getCrossAcceleration() const { return this->m_crossAccelertation; }
    uint16_t getCalculationMethod() const { return this->m_calculationMethod; }
    uint32_t getReserved2() const { return this->m_reserved2; }

public: // setter
    void setTimestamp(const NtpTime timestamp) { this->m_timestamp = timestamp; }
    void setScanNumber(const uint16_t scanNumber) { this->m_scanNumber = scanNumber; }
    void setErrorFlags(const uint16_t errorFlags) { this->m_errorFlags = errorFlags; }
    void setLongitudinalVelocity(const int16_t longitudinalVelocity)
    {
        this->m_longitudinalVelocity = longitudinalVelocity;
    }
    void setSteeringWheelAngle(const int16_t steeringWheelAngle) { this->m_steeringWheelAngle = steeringWheelAngle; }
    void setWheelAngle(const int16_t wheelAngle) { this->m_wheelAngle = wheelAngle; }
    void setXPosition(const int32_t xPos) { this->m_xPos = xPos; }
    void setYPosition(const int32_t yPos) { this->m_yPos = yPos; }
    void setCourseAngle(const int16_t courseAngle) { this->m_courseAngle = courseAngle; }
    void setTimeDiff(const uint16_t timeDiff) { this->m_timeDiff = timeDiff; }
    void setXDiff(const int16_t xDiff) { this->m_xDiff = xDiff; }
    void setYDiff(const int16_t yDiff) { this->m_yDiff = yDiff; }
    void setYaw(const int16_t yaw) { this->m_yaw = yaw; }
    void setCurrentYawRate(const int16_t currentYawRate) { this->m_currentYawRate = currentYawRate; }
    void setCrossAcceleration(const int16_t crossAcceleration) { this->m_crossAccelertation = crossAcceleration; }
    void setCalculationMethod(const uint16_t calcMethod) { this->m_calculationMethod = calcMethod; }
    void setReserved2(const uint32_t reserved2) { this->m_reserved2 = reserved2; }

protected:
    NtpTime m_timestamp{}; //!< The timestamp at which the vehicle had this state.
    uint16_t m_scanNumber{0}; //!< The Scan number of the scan this vehicle state is associated with.
    uint16_t m_errorFlags{0}; //!< Holds error flags in case of an error.
    int16_t m_longitudinalVelocity{0}; //!< Longitudinal Velocity (Car) [0.01 m/s]
    int16_t m_steeringWheelAngle{0}; //!< steering wheel angle [0.001 rad]
    int16_t m_wheelAngle{0}; //!<wheel angle (already converted from steering wheel angle if necessary) [0.0001 rad]
    int16_t m_crossAccelertation{0}; //!< cross acceleration [0.001 m/s²] // (was reserved0)

    // calculated (in DSP) movement of scanner

    // absolute values
    int32_t m_xPos{0}; //!< Absolute X Position from origin [0.01 m]
    int32_t m_yPos{0}; //!< Absolute Y Position from origin [0.01 m]
    int16_t m_courseAngle{0}; //!< Absolute orientation at time timeStamp [0.0001 rad]
    // relative values
    uint16_t m_timeDiff{0}; //!< Absolute orientation at time timeStamp [ms]
    int16_t m_xDiff{0}; //!< Difference in X during time difference to last transmission [0.001 m]
    int16_t m_yDiff{0}; //!< Difference in Y during time difference to last transmission [0.001 m]
    int16_t m_yaw{0}; //!< Difference in Heading during Timediff [0.0001 rad]
    // general informations

    //========================================
    //! \brief The Calculation method for motion estimation:
    //!
    //! \details 0=unknown,
    //!          bit1=SteeringWheelAngleUsed,
    //!          bit2=YawRateUsed
    //!
    //! (was reserved1)
    //----------------------------------------
    uint16_t m_calculationMethod{0};

    //========================================
    //! \brief Current yaw rate [0.0001rad/s] from latest CAN-Message. Since 2.5.00.
    //! \remark Available since firmware version 2.5.00.
    //----------------------------------------
    int16_t m_currentYawRate{0};

    uint32_t m_reserved2{0}; //!< unused, reserved for future use
}; // VehicleState2805

//==============================================================================

//==============================================================================

bool operator==(const VehicleState2805& lhs, const VehicleState2805& rhs);
bool operator!=(const VehicleState2805& lhs, const VehicleState2805& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
