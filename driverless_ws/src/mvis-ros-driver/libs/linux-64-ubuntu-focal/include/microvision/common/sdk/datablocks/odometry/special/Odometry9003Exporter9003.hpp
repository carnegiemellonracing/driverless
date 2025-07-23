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
//! \date Nov 21, 2019
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/odometry/special/Odometry9003.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Exporter for the Odometry9003 data container.
//------------------------------------------------------------------------------
template<>
class Exporter<Odometry9003, DataTypeId::DataType_Odometry9003>
  : public TypedExporter<Odometry9003, DataTypeId::DataType_Odometry9003>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //========================================
    //!\brief Convert to byte stream (serialization).
    //!
    //!\param[in, out] os  Output data stream.
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, \c false otherwise.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // Odometry9003Exporter9003

//==============================================================================

using Odometry9003Exporter9003 = Exporter<Odometry9003, DataTypeId::DataType_Odometry9003>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
