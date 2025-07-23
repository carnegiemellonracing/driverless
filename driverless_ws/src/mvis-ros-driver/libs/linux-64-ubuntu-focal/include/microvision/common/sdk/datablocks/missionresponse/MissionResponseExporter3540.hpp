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
//! \date May 27, 2019
//------------------------------------------------------------------------------
#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/missionresponse/MissionResponse.hpp>
#include <microvision/common/sdk/datablocks/missionresponse/special/MissionResponse3540.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//! \brief Mission Response Exporter
//!
//! Special data type: \ref microvision::common::sdk::MissionResponse3540
template<>
class Exporter<MissionResponse, DataTypeId::DataType_MissionResponse3540>
  : public TypedExporter<MissionResponse, DataTypeId::DataType_MissionResponse3540>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream.
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

}; //MissionResponseExporter3540

//==============================================================================

using MissionResponseExporter3540 = Exporter<MissionResponse, DataTypeId::DataType_MissionResponse3540>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
