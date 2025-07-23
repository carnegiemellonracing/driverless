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
//! \date March 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/timerelation/TimeRelation.hpp>

#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9010.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9011.hpp>

#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationIn9010.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationIn9011.hpp>

#include <microvision/common/sdk/datablocks/ClockType.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of relations for the reference timebase to an
//! 'Other' timebase
//!
//! Special data types:
//! \ref microvision::common::sdk::TimeRelationsList9010
//! \ref microvision::common::sdk::TimeRelationsList9011
//------------------------------------------------------------------------------
class TimeRelationsList final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using TimeRelationsMap = std::map<ClockType, TimeRelation>;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.timerelationslist"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    TimeRelationsList();
    TimeRelationsList(const TimeRelationsList9010& other);
    TimeRelationsList(const TimeRelationsList9011& other);
    TimeRelationsList& operator=(const TimeRelationsList&) = default;
    TimeRelationsList& operator                            =(const TimeRelationsList9010& other);
    TimeRelationsList& operator                            =(const TimeRelationsList9011& other);
    virtual ~TimeRelationsList()                           = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // setter
    void setRefTimeType(const ClockType refTimeType) { m_refTimeType = refTimeType; }

    void setTimeRelationsMap(const TimeRelationsMap& timeRelationsMap) { m_timeRelationsMap = timeRelationsMap; }

public: // getter
    const ClockType getRefTimeType() const { return m_refTimeType; }

public:
    // ========================================
    //! \brief Shorthand for getting the standard DUT clock.
    //! \return ClockType for DUT clock.
    //! \throws None Does not throw.
    // ----------------------------------------
    inline static ClockType getDutClock();

public: // accessors
    // ========================================
    //! \brief Determines whether the reference time is global, ie from GPS.
    //! \return \c True if the reference time is from GPS.
    //! \throw None Does not throw.
    //! \note The reference times are expressed in UTC in all cases.
    // ----------------------------------------
    bool isRefTimeGps() const;

    // ========================================
    //! \brief Determine if a TimeRelation exists with specified clock type.
    //! \param[in] type  ClockType to check against.
    //! \return \c True if there is a TimeRelation with the specified clock type.
    //! \throws None Does not throw.
    // ----------------------------------------
    bool hasTimeRelation(const ClockType type) const;

    // ========================================
    //! \brief Get the TimeRelation with specified clock type.
    //! \param[in] type  ClockType to use as index.
    //! \return The corresponding TimeRelation.
    //! \throws std::out_of_range If no such TimeRelation is present.
    // ----------------------------------------
    inline const TimeRelation& getTimeRelation(const ClockType type) const;

    // ========================================
    //! \brief Wrapper for \a getTimeRelation
    // ----------------------------------------
    inline const TimeRelation& operator[](const ClockType type) const;

    // ========================================
    //! \brief Get all contained TimeRelations.
    //! \return Map of contained TimeRelations
    //! \throws None Does not throw.
    // ----------------------------------------
    const TimeRelationsMap& getTimeRelations() const;

    // ========================================
    //! \brief Shorthand for getting the DUT TimeRelation
    // ----------------------------------------
    inline const TimeRelation& getDutTimeRelation() const;

public: // mutators
    // ========================================
    //! \brief Removes all contained TimeRelations.
    //! \throws None Does not throw.
    // ----------------------------------------
    void clear();

    // ========================================
    //! \brief Wrapper for \a getTimeRelation
    // ----------------------------------------
    TimeRelation& getTimeRelation(const ClockType type);

    // ========================================
    //! \brief Wrapper for \a getTimeRelation
    // ----------------------------------------
    inline TimeRelation& operator[](const ClockType type);

    // ========================================
    //! \brief Creates a TimeRelation for specified ClockType if none exists.
    //! \param[in] type  ClockType to use as index.
    //! \throws std::bad_alloc could be thrown if out of memory.
    // ----------------------------------------
    void ensureTimeRelationExists(const ClockType type);

public: // debugging functions
    std::string debugToString() const;

protected:
    ClockType m_refTimeType;
    TimeRelationsMap m_timeRelationsMap;

}; // TimeRelationsList

//==============================================================================

bool operator==(const TimeRelationsList& lhs, const TimeRelationsList& rhs);

//==============================================================================

bool operator!=(const TimeRelationsList& lhs, const TimeRelationsList& rhs);

//==============================================================================
// Inline function definitions

inline ClockType TimeRelationsList::getDutClock()
{
    const uint8_t defaultClockIDForCAN = static_cast<uint8_t>(0xFF);
    return ClockType(defaultClockIDForCAN, ClockType::ClockName::Dut);
}

//==============================================================================

const TimeRelation& TimeRelationsList::getTimeRelation(const ClockType type) const
{
    return const_cast<TimeRelationsList&>(*this).getTimeRelation(type);
}

//==============================================================================

TimeRelation& TimeRelationsList::operator[](const ClockType type) { return getTimeRelation(type); }

//==============================================================================

const TimeRelation& TimeRelationsList::operator[](const ClockType type) const { return getTimeRelation(type); }

//==============================================================================

const TimeRelation& TimeRelationsList::getDutTimeRelation() const { return getTimeRelation(getDutClock()); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
