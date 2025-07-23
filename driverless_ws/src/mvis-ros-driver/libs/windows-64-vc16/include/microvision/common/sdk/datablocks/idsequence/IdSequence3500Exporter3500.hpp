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
//! \date Sept 03, 2018
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/idsequence/IdSequence3500.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<IdSequence3500, DataTypeId::DataType_IdSequence3500>
  : public TypedExporter<IdSequence3500, DataTypeId::DataType_IdSequence3500>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; //IdSequence3500Exporter3500

//==============================================================================

using IdSequence3500Exporter3500 = Exporter<IdSequence3500, DataTypeId::DataType_IdSequence3500>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
