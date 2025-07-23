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
//! \date Mar 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/idctrailer/special/IdcTrailer6120.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::IdcTrailer6120, DataTypeId::DataType_IdcTrailer6120>
  : public TypedExporter<microvision::common::sdk::IdcTrailer6120, DataTypeId::DataType_IdcTrailer6120>
{
public:
    static constexpr std::streamsize serializedSize{0};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // IdcTrailer6120Exporter6120

//==============================================================================

using IdcTrailer6120Exporter6120
    = Exporter<microvision::common::sdk::IdcTrailer6120, DataTypeId::DataType_IdcTrailer6120>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
