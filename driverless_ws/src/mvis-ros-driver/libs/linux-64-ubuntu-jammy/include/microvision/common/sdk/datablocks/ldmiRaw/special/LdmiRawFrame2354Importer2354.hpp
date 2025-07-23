//==============================================================================
//! \file
//!
//! \brief Deserialize data container LdmiRawFrame2354 from stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 6th, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2354.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Deserialize data container LdmiRawFrame2354 from stream.
//------------------------------------------------------------------------------
template<>
class Importer<LdmiRawFrame2354, DataTypeId::DataType_LdmiRawFrame2354>
  : public RegisteredImporter<LdmiRawFrame2354, DataTypeId::DataType_LdmiRawFrame2354>
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    Importer() : RegisteredImporter() {}

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

}; // LdmiRawFrame2354Importer2354

//==============================================================================

using LdmiRawFrame2354Importer2354 = Importer<LdmiRawFrame2354, DataTypeId::DataType_LdmiRawFrame2354>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
