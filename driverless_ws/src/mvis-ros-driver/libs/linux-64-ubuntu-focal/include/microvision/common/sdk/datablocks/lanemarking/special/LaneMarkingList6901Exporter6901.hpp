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

#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingList6901.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This is the exporter for lane markings.
//!
//! Exports a laneMarkingList6901 into a Bytestream
//------------------------------------------------------------------------------
template<>
class Exporter<LaneMarkingList6901, static_cast<DataTypeId::DataType>(DataTypeId::DataType_LaneMarkingList6901)>
  : public TypedExporter<LaneMarkingList6901,
                         static_cast<DataTypeId::DataType>(DataTypeId::DataType_LaneMarkingList6901)>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

private:
    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const lanes::LaneMarkingIn6901& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const lanes::LaneMarkingSegmentIn6901& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const lanes::LaneMarkingPointIn6901& c);

}; // LaneMarkingList6901Exporter6901

//==============================================================================

using LaneMarkingList6901Exporter6901
    = Exporter<LaneMarkingList6901, static_cast<DataTypeId::DataType>(DataTypeId::DataType_LaneMarkingList6901)>;

//==============================================================================

template<>
void writeBE<lanes::LaneMarkingIn6901::LaneMarkingColor>(std::ostream& os,
                                                         const lanes::LaneMarkingIn6901::LaneMarkingColor& c);

//==============================================================================

template<>
void writeBE<lanes::LaneMarkingSegmentIn6901::LaneMarkingDashType>(
    std::ostream& os,
    const lanes::LaneMarkingSegmentIn6901::LaneMarkingDashType& c);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
