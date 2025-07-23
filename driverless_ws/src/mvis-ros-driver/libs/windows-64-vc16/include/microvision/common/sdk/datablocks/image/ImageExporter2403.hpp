//==============================================================================
//! \file
//!
//! Exporter for Image GDC to Image2403
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//!
//! Created by
//! \date June 18, 2018
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/image/Image.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<Image, DataTypeId::DataType_Image2403> : public TypedExporter<Image, DataTypeId::DataType_Image2403>
{
public:
    //========================================
    //!\brief get size in bytes of serialized data
    //!\param[in]      c       Data container.
    //!\return \c True if serialization succeed, else: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

    //!\brief convert to byte stream (serialization)
    //!\param[in, out] os      Output data stream
    //!\param[in]      c       Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // ImageExporter2403

//==============================================================================

using ImageExporter2403 = Exporter<Image, DataTypeId::DataType_Image2403>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
