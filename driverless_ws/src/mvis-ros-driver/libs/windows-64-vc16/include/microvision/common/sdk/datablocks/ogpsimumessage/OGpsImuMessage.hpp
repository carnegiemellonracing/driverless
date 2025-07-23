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
//! \date Jan 31, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/ogpsimumessage/special/OGpsImuMessage2610.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialisation of incoming third party OGpsImu UDP packets
//!
//! Special data type: \ref microvision::common::sdk::OGpsImuMessage2610
//------------------------------------------------------------------------------
class OGpsImuMessage final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const OGpsImuMessage& lhs, const OGpsImuMessage& rhs);

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.OGpsImuMessage"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    OGpsImuMessage()          = default;
    virtual ~OGpsImuMessage() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    uint32_t getIpAddress() const { return m_delegate.getIpAddress(); }
    const std::vector<uint8_t>& getData() const { return m_delegate.getData(); }
    std::vector<uint8_t>& getData() { return m_delegate.getData(); }

public: // setter
    void setIpAddress(const uint32_t ipAddress) { m_delegate.setIpAddress(ipAddress); }

private:
    OGpsImuMessage2610 m_delegate;
}; // OGpsImuMessage

//==============================================================================

inline bool operator==(const OGpsImuMessage& lhs, const OGpsImuMessage& rhs)
{
    return lhs.m_delegate == rhs.m_delegate;
}

inline bool operator!=(const OGpsImuMessage& lhs, const OGpsImuMessage& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
