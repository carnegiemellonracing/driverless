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
//! \date Feb 02, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationList7110.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<MetaInformationList7110, DataTypeId::DataType_MetaInformationList7110>
  : public TypedExporter<MetaInformationList7110, DataTypeId::DataType_MetaInformationList7110>
{
public:
    virtual ~Exporter() = default;

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // MetaInformationList7110Exporter7110

//==============================================================================

using MetaInformationList7110Exporter7110
    = Exporter<microvision::common::sdk::MetaInformationList7110, DataTypeId::DataType_MetaInformationList7110>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
