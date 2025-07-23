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
//! \date Jul 30, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2321.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to export a third party lidar raw data scan (data type 2321)
//!        into an IDC container of the same type.
//------------------------------------------------------------------------------
template<>
class Exporter<Scan2321, DataTypeId::DataType_Scan2321> : public TypedExporter<Scan2321, DataTypeId::DataType_Scan2321>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // Scan2321Exporter2321

//==============================================================================

using Scan2321Exporter2321 = Exporter<Scan2321, DataTypeId::DataType_Scan2321>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
