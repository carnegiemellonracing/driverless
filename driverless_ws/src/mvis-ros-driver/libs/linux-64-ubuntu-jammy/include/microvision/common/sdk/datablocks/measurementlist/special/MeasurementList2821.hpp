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
//! \date Jan 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementSeriesIn2821.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU measurement list:
//! Measurement List (generic)
//!
//! Measurement List data to represent dynamic measurements provided by an sensor system.
//!
//! All angles, position and distances are given in the ISO 8855 / DIN 70000 coordinate system.
//------------------------------------------------------------------------------
class MeasurementList2821 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.measurementlist2821"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    MeasurementList2821();
    ~MeasurementList2821() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint32_t getMicroseconds() const { return m_microseconds; }
    NtpTime getTimestamp() const { return m_timestamp; }

    const std::string& getListName() const { return m_listName; }
    std::string& getListName() { return m_listName; }
    const std::string& getGroupName() const { return m_groupName; }
    std::string& getGroupName() { return m_groupName; }
    const MeasurementSeriesIn2821& getMeasList() const { return m_measurements; }
    MeasurementSeriesIn2821& getMeasList() { return m_measurements; }

public:
    void setMicroseconds(const uint32_t newMicroseconds) { m_microseconds = newMicroseconds; }
    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }
    void setListName(const std::string& newListName) { m_listName = newListName; }
    void setGroupName(const std::string& newGroupName) { m_groupName = newGroupName; }
    void setMeasurements(const MeasurementSeriesIn2821& newMeasurementList) { m_measurements = newMeasurementList; }

protected:
    uint32_t m_microseconds{0}; //!< Microseconds since power-on.
    NtpTime m_timestamp{}; //!< Timestamp of this measurements.
    std::string m_listName{}; //!< Name of the measurement list. Used to identify source of the measurements.
    std::string m_groupName{}; //!< Intended for group selection.
    MeasurementSeriesIn2821 m_measurements{}; //!< Vector of Measurements.
}; // MeasurementList2821Container

//==============================================================================

bool operator==(const MeasurementList2821& lhs, const MeasurementList2821& rhs);
inline bool operator!=(const MeasurementList2821& lhs, const MeasurementList2821& rhs) { return !(rhs == lhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
