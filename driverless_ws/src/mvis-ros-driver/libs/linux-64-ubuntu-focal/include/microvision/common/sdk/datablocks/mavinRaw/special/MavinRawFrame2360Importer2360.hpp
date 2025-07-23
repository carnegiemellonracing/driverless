//==============================================================================
//! \file
//!
//! \brief Importer for MavinRawFrame2360 from binary format for MavinRawFrame2360.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 1th, 2023
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/mavinRaw/special/MavinRawFrame2360.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Deserialize data container MavinRawFrame2360 from stream.
//------------------------------------------------------------------------------
template<>
class Importer<MavinRawFrame2360, DataTypeId::DataType_MavinRawFrame2360>
  : public RegisteredImporter<MavinRawFrame2360, DataTypeId::DataType_MavinRawFrame2360>
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

}; // MavinRawFrame2360Importer2360

//==============================================================================

using MavinRawFrame2360Importer2360 = Importer<MavinRawFrame2360, DataTypeId::DataType_MavinRawFrame2360>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
