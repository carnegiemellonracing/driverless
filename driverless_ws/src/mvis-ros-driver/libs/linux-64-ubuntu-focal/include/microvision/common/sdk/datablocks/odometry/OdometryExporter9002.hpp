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
//! \date Mar 18, 2018
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/odometry/Odometry.hpp>
#include <microvision/common/sdk/datablocks/odometry/special/Odometry9002.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<Odometry, DataTypeId::DataType_Odometry9002>
  : public TypedExporter<Odometry, DataTypeId::DataType_Odometry9002>
{
public:
    static constexpr std::streamsize serializedBaseSize{108};

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // OdometryExporter9002

//==============================================================================

using OdometryExporter9002 = Exporter<Odometry, DataTypeId::DataType_Odometry9002>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
