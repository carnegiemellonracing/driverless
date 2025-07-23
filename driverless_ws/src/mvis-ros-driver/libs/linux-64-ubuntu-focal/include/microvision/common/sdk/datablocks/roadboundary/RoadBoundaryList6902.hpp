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
//! \date Okt 19, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryIn6902.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of road boundaries.
//!
//! There is currently no general data type.
//------------------------------------------------------------------------------
class RoadBoundaryList6902 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.RoadBoundaryList6902"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Vector of road boundaries in this list.
    //----------------------------------------
    using RoadBoundaries = std::vector<RoadBoundaryIn6902>;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    RoadBoundaryList6902() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~RoadBoundaryList6902() override = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Get the vector of the road boundaries.
    //! \return The vector of road boundaries.
    //----------------------------------------
    const RoadBoundaries& getRoadBoundaries() const { return m_roadBoundaries; }

    //========================================
    //! \brief Get the vector of the road boundaries.
    //! \return The vector of road boundaries.
    //----------------------------------------
    RoadBoundaries& getRoadBoundaries() { return m_roadBoundaries; }

    //========================================
    //! \brief Sets the vector of the road boundaries.
    //! \param[in] boundaries  The new vector.
    //----------------------------------------
    void setRoadBoundaries(const RoadBoundaries& boundaries) { m_roadBoundaries = boundaries; }

    //========================================
    //! \brief Adds a road boundary at the end of the vector of road boundaries.
    //! \param[in] boundary The added boundary.
    //----------------------------------------
    void addRoadBoundary(const RoadBoundaryIn6902& boundary) { m_roadBoundaries.push_back(boundary); }

private:
    RoadBoundaries m_roadBoundaries{};
}; // RoadBoundaryList6902

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const RoadBoundaryList6902& lhs, const RoadBoundaryList6902& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const RoadBoundaryList6902& lhs, const RoadBoundaryList6902& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
