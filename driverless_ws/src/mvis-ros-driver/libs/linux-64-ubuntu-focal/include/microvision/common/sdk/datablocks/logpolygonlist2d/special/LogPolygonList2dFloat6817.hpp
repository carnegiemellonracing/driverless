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
//! \date Mar 23, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/logpolygonlist2d/LogPolygon2d.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of informational polygons with text label
//!
//! General data type: \ref microvision::common::sdk::LogPolygonList2d
//------------------------------------------------------------------------------
class LogPolygonList2dFloat6817 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using LogPolygonType       = LogPolygon2d<float>;
    using LogPolygonListVector = std::vector<LogPolygonType>;
    using iterator             = LogPolygonListVector::iterator;
    using const_iterator       = LogPolygonListVector::const_iterator;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.logpolygonlist2dfloat6817"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    LogPolygonList2dFloat6817() : SpecializedDataContainer{} {}
    virtual ~LogPolygonList2dFloat6817() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    const LogPolygonListVector& getLogList() const { return m_logList; }
    void setLogList(const LogPolygonListVector& logList) { m_logList = logList; }

private:
    LogPolygonListVector m_logList;
}; // LogPolygonList2dFloat6817

//==============================================================================

inline bool operator==(const LogPolygonList2dFloat6817& lhs, const LogPolygonList2dFloat6817& rhs)
{
    return (lhs.getLogList() == rhs.getLogList());
}

inline bool operator!=(const LogPolygonList2dFloat6817& lhs, const LogPolygonList2dFloat6817& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
