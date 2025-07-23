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
//! \date Nov 6th, 2019
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2404.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! Exporter for specialized image data container 0x2404 to binary data
//! in stream.
//------------------------------------------------------------------------------
template<>
class Exporter<microvision::common::sdk::Image2404, DataTypeId::DataType_Image2404>
  : public TypedExporter<microvision::common::sdk::Image2404, DataTypeId::DataType_Image2404>
{
public:
    //========================================
    //! \brief Get the serialized size in bytes for this image.
    //!
    //! \param[in] c  Data container to be serialized.
    //! \return Size of serialized data container in bytes.
    //========================================
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    //========================================
    //! \brief Write data container to serialized binary data stream (serialization).
    //!
    //! \param[in, out] os          Output data stream
    //! \param[in] importContainer  Input data container.
    //! \return \c True if serialization succeeded, \c false if not.
    //========================================
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // Image2404Exporter2404

//==============================================================================

using Image2404Exporter2404 = Exporter<microvision::common::sdk::Image2404, DataTypeId::DataType_Image2404>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
