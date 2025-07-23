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
//! \date Jan 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU GPS position:
//! WGS84 position w/ device ID
//------------------------------------------------------------------------------
class PositionWgs84_2604 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    PositionWgs84_2604();
    virtual ~PositionWgs84_2604();

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.positionwgs84_2604"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    const PositionWgs84& getPosition() const { return m_positionWgs84; }
    PositionWgs84& getPosition() { return m_positionWgs84; }

protected:
    PositionWgs84 m_positionWgs84;
}; // PositionWgs84_2604

//==============================================================================

bool operator==(const PositionWgs84_2604& lhs, const PositionWgs84_2604& rhs);

//==============================================================================

inline bool operator!=(const PositionWgs84_2604& lhs, const PositionWgs84_2604& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
