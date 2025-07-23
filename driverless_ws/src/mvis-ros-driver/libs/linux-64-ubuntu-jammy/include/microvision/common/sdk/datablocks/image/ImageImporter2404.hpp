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
//! \date Nov 7th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/image/Image.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Importer for Image from 2404.
//!
//! Importer for general data container type Image from serialized binary data
//! stream of specialized 2404 data container type.
//------------------------------------------------------------------------------
template<>
class Importer<Image, DataTypeId::DataType_Image2404> : public RegisteredImporter<Image, DataTypeId::DataType_Image2404>
{
public:
    //========================================
    //! Constructor calling base class constructor.
    //----------------------------------------
    Importer() : RegisteredImporter() {}

    //========================================
    //! \brief Deleted copy constructor.
    //!
    //! Importers can not be copied.
    //----------------------------------------
    Importer(const Importer&) = delete;

    //========================================
    //! \brief Deleted assignment operator.
    //!
    //! Importers can not be copied.
    //----------------------------------------
    Importer& operator=(const Importer&) = delete;

public: // implements ImporterBase
    //=================================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //-------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                      const ConfigurationPtr& configuration = nullptr) const override;

    //=================================================
    //! \brief Read data from the given stream and fill the given data container (deserialization).
    //!
    //! \param[in, out] inputStream     Input data stream
    //! \param[out]     dataContainer   Output container defining the target type (might include conversion).
    //! \param[in]      dataHeader      Metadata prepended to each idc data block.
    //! \param[in]      configuration   (Optional) Configuration context for import. Default set as nullptr.
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //-------------------------------------------------
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;
};

//==============================================================================

using ImageImporter2404 = Importer<microvision::common::sdk::Image, DataTypeId::DataType_Image2404>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
