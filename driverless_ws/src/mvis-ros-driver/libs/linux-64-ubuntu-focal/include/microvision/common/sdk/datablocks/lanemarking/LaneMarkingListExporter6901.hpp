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
//! \date Nov 20, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

#include <microvision/common/sdk/datablocks/lanemarking/LaneMarkingList.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This is the exporter for lane markings.
//------------------------------------------------------------------------------
template<>
class Exporter<LaneMarkingList, static_cast<DataTypeId::DataType>(DataTypeId::DataType_LaneMarkingList6901)>
  : public TypedExporter<LaneMarkingList, static_cast<DataTypeId::DataType>(DataTypeId::DataType_LaneMarkingList6901)>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //!\note This method is to be called from outside for serialization.
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LaneMarkingListExporter6901

//==============================================================================

using LaneMarkingListExporter6901
    = Exporter<LaneMarkingList, static_cast<DataTypeId::DataType>(DataTypeId::DataType_LaneMarkingList6901)>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
