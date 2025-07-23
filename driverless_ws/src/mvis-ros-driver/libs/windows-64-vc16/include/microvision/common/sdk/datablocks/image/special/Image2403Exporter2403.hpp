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
//! \date Mar 9, 2018
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2403.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::Image2403, DataTypeId::DataType_Image2403>
  : public TypedExporter<microvision::common::sdk::Image2403, DataTypeId::DataType_Image2403>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // Image2403Exporter2403

//==============================================================================

using Image2403Exporter2403 = Exporter<microvision::common::sdk::Image2403, DataTypeId::DataType_Image2403>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
