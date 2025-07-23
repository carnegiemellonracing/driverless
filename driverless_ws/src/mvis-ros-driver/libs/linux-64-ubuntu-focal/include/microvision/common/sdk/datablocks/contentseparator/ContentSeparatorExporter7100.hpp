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
//! \date Feb 4, 2019
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/contentseparator/ContentSeparator.hpp>
#include <microvision/common/sdk/datablocks/contentseparator/special/ContentSeparator7100.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<ContentSeparator, DataTypeId::DataType_ContentSeparator7100>
  : public TypedExporter<ContentSeparator, DataTypeId::DataType_ContentSeparator7100>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // ContentSeparatorExporter7100

//==============================================================================

using ContentSeparatorExporter7100 = Exporter<ContentSeparator, DataTypeId::DataType_ContentSeparator7100>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
