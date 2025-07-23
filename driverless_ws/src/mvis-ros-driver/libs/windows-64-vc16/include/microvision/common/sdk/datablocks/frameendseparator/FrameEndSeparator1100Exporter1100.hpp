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
//! \date Jan 11, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/frameendseparator/FrameEndSeparator1100.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<FrameEndSeparator1100, DataTypeId::DataType_FrameEndSeparator1100>
  : public TypedExporter<FrameEndSeparator1100, DataTypeId::DataType_FrameEndSeparator1100>
{
public:
    static constexpr std::streamsize serializedSize{32};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // FrameEndSeparator1100Exporter1100

//==============================================================================

using FrameEndSeparator1100Exporter1100 = Exporter<FrameEndSeparator1100, DataTypeId::DataType_FrameEndSeparator1100>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
