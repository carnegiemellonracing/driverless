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
//! \date Mar 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/eventmarker/special/EventMarker7001.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<EventMarker7001, DataTypeId::DataType_EventMarker7001>
  : public TypedExporter<EventMarker7001, DataTypeId::DataType_EventMarker7001>
{
public:
    static constexpr std::streamsize serializedBaseSize{10};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // EventMarker7001Exporter7001

//==============================================================================

using EventMarker7001Exporter7001 = Exporter<EventMarker7001, DataTypeId::DataType_EventMarker7001>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
