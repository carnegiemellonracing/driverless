//==============================================================================
//! Exporter for Image GDC to Image2405
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 7th, 2019
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

//==============================================================================
//! \brief Image to 2405 exporter.
//!
//! Exporter for general data container type Image as serialized binary data stream
//! of specialized 2405 data container type.
//------------------------------------------------------------------------------
template<>
class Exporter<Image, DataTypeId::DataType_Image2405> : public TypedExporter<Image, DataTypeId::DataType_Image2405>
{
public:
    //========================================
    //! \brief Get size of the serialized data in bytes.
    //! \param[in] c  Data container.
    //! \return Size of serialized binary data in bytes.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //! \brief Output image data container to byte stream (serialization).
    //! \param[in, out] os               Output data stream
    //! \param[in]      importContainer  Data container.
    //! \return \c True if serialization succeeded, \c false if not.
    //----------------------------------------
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // ImageExporter2405

//==============================================================================

using ImageExporter2405 = Exporter<Image, DataTypeId::DataType_Image2405>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
