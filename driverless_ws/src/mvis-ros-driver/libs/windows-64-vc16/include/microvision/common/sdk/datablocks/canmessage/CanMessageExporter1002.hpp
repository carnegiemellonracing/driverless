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
//! \date 12.October 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/canmessage/CanMessage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<CanMessage, DataTypeId::DataType_CanMessage1002>
  : public TypedExporter<CanMessage, DataTypeId::DataType_CanMessage1002>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& outStream, const DataContainerBase& c) const override;
}; //CanMessageExporter1002

//==============================================================================

using CanMessageExporter1002 = Exporter<CanMessage, DataTypeId::DataType_CanMessage1002>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
