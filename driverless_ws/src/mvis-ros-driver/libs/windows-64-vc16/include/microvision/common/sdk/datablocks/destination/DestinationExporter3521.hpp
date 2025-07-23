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
//! \date Jan 28, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/destination/Destination.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Exports general Destination data container as Destination3521.
//------------------------------------------------------------------------------
template<>
class Exporter<Destination, DataTypeId::DataType_Destination3521>
  : public TypedExporter<Destination, DataTypeId::DataType_Destination3521>
{
public:
    //==============================================================================
    //!\brief get size in bytes of serialized data
    //!\param[in] c  Data container.
    //!\return \c True if serialization succeed, else: \c false
    //------------------------------------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //==============================================================================
    //!\brief convert to byte stream (serialization)
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    //------------------------------------------------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; //DestinationExporter3521

//==============================================================================

using DestinationExporter3521 = Exporter<Destination, DataTypeId::DataType_Destination3521>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
