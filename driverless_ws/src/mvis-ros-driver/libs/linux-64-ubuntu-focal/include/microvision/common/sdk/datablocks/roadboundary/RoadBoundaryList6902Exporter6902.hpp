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
//! \date Okt 19, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryList6902.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This is the exporter for road boundaries.
//!
//! Serializes a RoadBoundaryList6902 into a Bytestream.
//------------------------------------------------------------------------------
template<>
class Exporter<RoadBoundaryList6902, static_cast<DataTypeId::DataType>(DataTypeId::DataType_RoadBoundaryList6902)>
  : public TypedExporter<RoadBoundaryList6902,
                         static_cast<DataTypeId::DataType>(DataTypeId::DataType_RoadBoundaryList6902)>
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
    //!\param[in]      boundary   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const RoadBoundaryIn6902& boundary);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const RoadBoundarySegmentIn6902& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const RoadBoundaryPointIn6902& c);

}; // RoadBoundaryList6902Exporter6902

//==============================================================================

using RoadBoundaryList6902Exporter6902
    = Exporter<RoadBoundaryList6902, static_cast<DataTypeId::DataType>(DataTypeId::DataType_RoadBoundaryList6902)>;

//==============================================================================

template<>
void writeBE<RoadBoundaryIn6902::RoadBoundaryColor>(std::ostream& os, const RoadBoundaryIn6902::RoadBoundaryColor& c);

//==============================================================================

template<>
void writeBE<RoadBoundarySegmentIn6902::RoadBoundaryDashType>(std::ostream& os,
                                                              const RoadBoundarySegmentIn6902::RoadBoundaryDashType& c);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
