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
//! \date May 23, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/eventmarker/EventMarker.hpp>
#include <microvision/common/sdk/datablocks/eventmarker/special/EventMarker7001.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Event marker exporter
//!
//! Special data type: \ref microvision::common::sdk::EventMarker7001
//------------------------------------------------------------------------------
template<>
class Exporter<EventMarker, DataTypeId::DataType_EventMarker7001>
  : public TypedExporter<EventMarker, DataTypeId::DataType_EventMarker7001>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //!\brief convert to byte stream (serialization)
    //!\param[in, out] os Output data stream
    //!\param[in]      c  Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // EventMarkerExporter7001

//==============================================================================

using EventMarkerExporter7001 = Exporter<EventMarker, DataTypeId::DataType_EventMarker7001>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
