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
//! \date Mar 21, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/timerecord/special/TimeRecord9000.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<TimeRecord9000, DataTypeId::DataType_TimeRecord9000>
  : public TypedExporter<TimeRecord9000, DataTypeId::DataType_TimeRecord9000>
{
public:
    Exporter() = default;

    Exporter(const Exporter&) = delete;
    Exporter& operator=(const Exporter&) = delete;

public:
    static constexpr std::streamsize serializedBaseSize{20};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // TimeRecord9000Exporter9000

//==============================================================================

using TimeRecord9000Exporter9000 = Exporter<TimeRecord9000, DataTypeId::DataType_TimeRecord9000>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
