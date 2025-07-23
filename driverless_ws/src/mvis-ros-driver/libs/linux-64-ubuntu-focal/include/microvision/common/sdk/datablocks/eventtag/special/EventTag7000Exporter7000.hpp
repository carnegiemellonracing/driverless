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
#include <microvision/common/sdk/datablocks/eventtag/special/EventTag7000.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<EventTag7000, DataTypeId::DataType_EventTag7000>
  : public TypedExporter<EventTag7000, DataTypeId::DataType_EventTag7000>
{
public:
    static constexpr std::streamsize serializedBaseSize{40 + EventTag7000::nbOfReserved};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // EventTag7000Exporter7000

//==============================================================================

using EventTag7000Exporter7000 = Exporter<EventTag7000, DataTypeId::DataType_EventTag7000>;
//==============================================================================

template<>
void writeBE<EventTag7000::TagClass>(std::ostream& os, const EventTag7000::TagClass& tc);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
