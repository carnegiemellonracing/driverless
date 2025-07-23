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
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/datablocks/vehiclerequest/special/VehicleRequest9105.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Request sent to a VCU (vehicle control unit)
//!
//! Special data type: \ref microvision::common::sdk::VehicleRequest9105
//------------------------------------------------------------------------------
class VehicleRequest final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.vehiclerequest"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    VehicleRequest();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~VehicleRequest() = default;

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
    void setTimestamp(const Timestamp& timestamp) { m_delegate.setTimestamp(timestamp); }

    //========================================
    //!\brief Set the requestId of the VehicleRequest
    //!\param[in] requestId  Indicates what is being requested
    //----------------------------------------
    void setRequestId(const VehicleRequest9105::RequestId requestId) { m_delegate.setRequestId(requestId); }

    //========================================
    //!\brief Set the requestValue of the VehicleRequest
    //!\param[in] requestValue  The value being requested
    //----------------------------------------
    void setRequestValue(const VehicleRequest9105::RequestValue requestValue)
    {
        m_delegate.setRequestValue(requestValue);
    }

public:
    //========================================
    //!\brief Get the timestamp of the VehicleRequest
    //!\return timestamp of the VehicleRequest
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_delegate.getTimestamp(); }

    //========================================
    //!\brief Get the requestId of the VehicleRequest
    //!\return requestId of the VehicleRequest
    //----------------------------------------
    VehicleRequest9105::RequestId getRequestId() const { return m_delegate.getRequestId(); }

    //========================================
    //!\brief Get the requestValue of the VehicleRequest
    //!\return requestValue of the VehicleRequest
    //----------------------------------------
    VehicleRequest9105::RequestValue getRequestValue() const { return m_delegate.getRequestValue(); }

protected:
    VehicleRequest9105 m_delegate; //!< only possible specialization currently
}; // VehicleRequest

//==============================================================================

bool operator==(const VehicleRequest& lhs, const VehicleRequest& rhs);
bool operator!=(const VehicleRequest& lhs, const VehicleRequest& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
