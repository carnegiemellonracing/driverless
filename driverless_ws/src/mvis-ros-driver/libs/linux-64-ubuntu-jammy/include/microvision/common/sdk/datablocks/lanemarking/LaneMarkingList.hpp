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
//! \date Nov 19, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/lanemarking/LaneMarking.hpp>
#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingList6901.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of lane marking containing a list of lane marking segments.
//!
//! Used for generating lanes.
//!
//! Special data type: \ref microvision::common::sdk::LaneMarkingList6973
//------------------------------------------------------------------------------
class LaneMarkingList : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.lanemarkinglist"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Vector of lane markings in this list..
    //----------------------------------------
    using LaneMarkings = std::vector<lanes::LaneMarking>;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    LaneMarkingList() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LaneMarkingList() override = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Get the vector of the lane markings.
    //! \return The vector of lane markings.
    //----------------------------------------
    const LaneMarkings& getLaneMarkings() const { return m_delegate.getLaneMarkings(); }

    //========================================
    //! \brief Get the vector of the lane markings.
    //! \return The vector of lane markings.
    //----------------------------------------
    LaneMarkings& getLaneMarkings() { return m_delegate.getLaneMarkings(); }

    //========================================
    //! \brief Sets the vector of the lane markings.
    //! \param[in] markings  The new vector.
    //----------------------------------------
    void setLaneMarkings(const LaneMarkings& markings) { m_delegate.setLaneMarkings(markings); }

    //========================================
    //! \brief Adds a lane marking at the end of the vector of lane markings.
    //! \param[in] marking  The added marking.
    //----------------------------------------
    void addLaneMarking(const lanes::LaneMarkingIn6901& marking) { m_delegate.addLaneMarking(marking); }

private:
    LaneMarkingList6901 m_delegate; // only possible specialization currently
}; // LaneMarkingList

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const LaneMarkingList& lhs, const LaneMarkingList& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const LaneMarkingList& lhs, const LaneMarkingList& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
