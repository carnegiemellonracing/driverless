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
//! \date Jan 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/contentseparator/special/ContentSeparator7100.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::ContentSeparator7100, DataTypeId::DataType_ContentSeparator7100>
  : public TypedExporter<microvision::common::sdk::ContentSeparator7100, DataTypeId::DataType_ContentSeparator7100>
{
public:
    static constexpr std::streamsize serializedSize{std::streamsize(sizeof(uint16_t) + sizeof(uint32_t))};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // ContentSeparator7100Exporter7100

//==============================================================================

using ContentSeparator7100Exporter7100
    = Exporter<microvision::common::sdk::ContentSeparator7100, DataTypeId::DataType_ContentSeparator7100>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
