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
//! \date Jan 23, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/logpolygonlist2d/LogPolygonList2d.hpp>
#include <microvision/common/sdk/datablocks/logpolygonlist2d/special/LogPolygonList2dFloat6817.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of informational polygons with text label
//!
//! Special data types: \ref microvision::common::sdk::LogPolygonList2dFloat6817
//------------------------------------------------------------------------------
class LogPolygonList2d final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const LogPolygonList2d& lhs, const LogPolygonList2d& rhs);

public:
    using LogPolygonType       = LogPolygon2d<float>;
    using LogPolygonListVector = std::vector<LogPolygonType>;
    using iterator             = LogPolygonListVector::iterator;
    using const_iterator       = LogPolygonListVector::const_iterator;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.logpolygonlist2d"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    LogPolygonList2d() : DataContainerBase{} {}
    virtual ~LogPolygonList2d() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    const LogPolygonListVector& getLogList() const { return m_delegate.getLogList(); }
    void setLogList(const LogPolygonListVector& logList) { m_delegate.setLogList(logList); }

private:
    LogPolygonList2dFloat6817 m_delegate;
}; // LogPolygonList2d

//==============================================================================

inline bool operator==(const LogPolygonList2d& lhs, const LogPolygonList2d& rhs)
{
    return (lhs.m_delegate == rhs.m_delegate);
}

inline bool operator!=(const LogPolygonList2d& lhs, const LogPolygonList2d& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
