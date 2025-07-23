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
//! \date Jan 25, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include "Destination3521.hpp"

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Imports from Destination3521 to Destination3521 data container.
//------------------------------------------------------------------------------
template<>
class Importer<Destination3521, DataTypeId::DataType_Destination3521>
  : public RegisteredImporter<Destination3521, DataTypeId::DataType_Destination3521>
{
public:
    //==============================================================================
    //! \brief Constructor.
    //------------------------------------------------------------------------------
    Importer() : RegisteredImporter() {}

    //==============================================================================
    //! \brief Deleted copy constructor.
    //------------------------------------------------------------------------------
    Importer(const Importer&) = delete;

    //==============================================================================
    //! \brief Deleted copy assignment operator.
    //------------------------------------------------------------------------------
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
    //! \return \c true if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //-------------------------------------------------
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;

}; //Destination3521Importer3521

//==============================================================================

using Destination3521Importer3521 = Importer<Destination3521, DataTypeId::DataType_Destination3521>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
