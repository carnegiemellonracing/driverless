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

#include <microvision/common/sdk/datablocks/frameindex/FrameIndex.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::FrameIndex, DataTypeId::DataType_FrameIndex6130>
  : public TypedExporter<microvision::common::sdk::FrameIndex, DataTypeId::DataType_FrameIndex6130>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // FrameIndexExporter6130

//==============================================================================

using FrameIndexExporter6130 = Exporter<microvision::common::sdk::FrameIndex, DataTypeId::DataType_FrameIndex6130>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
