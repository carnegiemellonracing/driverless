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
//! \date Jan 25, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include "Destination3521.hpp"

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Exports Destination3521 data container as Destination3521.
//------------------------------------------------------------------------------
template<>
class Exporter<Destination3521, DataTypeId::DataType_Destination3521>
  : public TypedExporter<Destination3521, DataTypeId::DataType_Destination3521>
{
public:
    //==============================================================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //------------------------------------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //==============================================================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream.
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, \c false otherwise.
    //------------------------------------------------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; //Destination3521

//==============================================================================

using Destination3521Exporter3521 = Exporter<Destination3521, DataTypeId::DataType_Destination3521>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
