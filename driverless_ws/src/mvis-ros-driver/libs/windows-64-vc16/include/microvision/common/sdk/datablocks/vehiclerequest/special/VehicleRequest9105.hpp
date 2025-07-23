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
//! \date Nov 6, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Request sent to a VCU (vehicle control unit)
//!
//! General data type: \ref microvision::common::sdk::VehicleRequest
//------------------------------------------------------------------------------
class VehicleRequest9105 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using RequestValue = int8_t; //!< Type for the value being requested

public:
    //========================================
    //!\enum RequestId
    //! Identifier indicating what is being requested.
    //----------------------------------------
    enum class RequestId : uint8_t
    {
        StopIntention         = 0, //!< Intent to stop the vehicle
        GearRequest           = 1, //!< Request gear change (see \ref GearState)
        VehicleReleaseRequest = 2, //!< In driving mode passive, request clearance
        EmergencyBrakeRequest = 3, //!< Request an emergency brake
        HighBeamRequest       = 4, //!< Request high beam
        LowBeamRequest        = 5, //!< Request low beam
        IndicatorRequest      = 6, //!< Request indicator (see \ref IndicatorState)
        HornRequest           = 7, //!< Request horn
        HandoverSoftRequest   = 8, //!< Request soft takeover to driver
        HandoverHardRequest   = 9, //!< Request hard takeover to driver
        Unknown               = 0xFF //!< Unknown or invalid request
    };

    //========================================
    //!\enum IndicatorState
    //! Type for different indicator states
    //----------------------------------------
    enum class IndicatorState : int8_t
    {
        Off   = 0, //!< Set indicator to off
        Left  = 1, //!< Activate left indicator
        Right = 2, //!< Activate right indicator
        Both  = 3 //!< Activate both indicators
    };

    //========================================
    //!\enum  Gear
    //! Type for different vehicle gear states
    //----------------------------------------
    enum class GearState : int8_t
    {
        Parking = -4, //!< Parking position
        Reverse = -3, //!< Reverse driving
        Sport   = -2, //!< Sport sport shifting
        Driving = -1, //!< Driving gear
        Neutral = 0, //!< Neutral gear
        First   = 1, //!< First manual gear
        Second  = 2, //!< Second manual gear
        Third   = 3, //!< Third manual gear
        Fourth  = 4, //!< Fourth manual gear
        Fifth   = 5, //!< Firth manual gear
        Sixth   = 6 //!< Sixth manual gear
    };

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.vehiclerequest9105"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    VehicleRequest9105() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~VehicleRequest9105() = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Set the timestamp of the VehicleRequest
    //!\param[in] timestamp  New timestamp of the VehicleRequest
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { this->m_timestamp = timestamp; }

    //========================================
    //!\brief Set the requestId of the VehicleRequest
    //!\param[in] requestId  Indicates what is being requested
    //----------------------------------------
    void setRequestId(const RequestId requestId) { this->m_requestId = requestId; }

    //========================================
    //!\brief Set the requestValue of the VehicleRequest
    //!\param[in] requestValue  The value being requested
    //----------------------------------------
    void setRequestValue(const RequestValue requestValue) { this->m_requestValue = requestValue; }

    //========================================
    //!\brief Get the timestamp of the VehicleRequest
    //!\return timestamp of the VehicleRequest
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_timestamp; }

    //========================================
    //!\brief Get the requestId of the VehicleRequest
    //!\return requestId of the VehicleRequest
    //----------------------------------------
    RequestId getRequestId() const { return m_requestId; }

    //========================================
    //!\brief Get the requestValue of the VehicleRequest
    //!\return requestValue of the VehicleRequest
    //----------------------------------------
    RequestValue getRequestValue() const { return m_requestValue; }

protected:
    Timestamp m_timestamp{}; //!< Timestamp of VehicleRequest
    RequestId m_requestId{RequestId::Unknown}; //!< What is being requested
    RequestValue m_requestValue{0}; //!< Value for request (true/false/enum)
}; // VehicleRequest9105

//==============================================================================

bool operator==(const VehicleRequest9105& lhs, const VehicleRequest9105& rhs);
bool operator!=(const VehicleRequest9105& lhs, const VehicleRequest9105& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
