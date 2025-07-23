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
//! \date Apr 4, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayTemplate.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayIn6970.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of carriageways containing a list of lane segments.
//!
//! General data type: \ref microvision::common::sdk::CarriageWayList
//------------------------------------------------------------------------------
class CarriageWayList6970 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.carriagewaylist6970"};

    using CarriageWays = std::vector<typename lanes::CarriageWayTemplate<lanes::LaneSegmentIn6970>::Ptr>;

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CarriageWayList6970();
    virtual ~CarriageWayList6970() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    const CarriageWays& getCarriageWays() const { return m_carriageWays; }
    CarriageWays& getCarriageWays() { return m_carriageWays; }

private:
    CarriageWays m_carriageWays{};
}; // CarriageWayList6970

//==============================================================================

bool operator==(const CarriageWayList6970& lhs, const CarriageWayList6970& rhs);

inline bool operator!=(const CarriageWayList6970& lhs, const CarriageWayList6970& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
