//==============================================================================
//! \file
//!
//! \brief Imports object type 0x2280 from general object container
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 12, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectList.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2280_2290.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<ObjectList, DataTypeId::DataType_ObjectList2280>
  : public RegisteredImporter<ObjectList, DataTypeId::DataType_ObjectList2280>, protected ObjectListImporter2280_2290
{
public:
    virtual ~Importer() = default;

public:
public: // implements ImporterBase
    //=================================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //-------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                      const ConfigurationPtr& configuration = nullptr) const override
    {
        (void)configuration;
        return ObjectListImporter2280_2290::getSerializedSize(dataContainer);
    }

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
                     const ConfigurationPtr& configuration = nullptr) const override
    {
        (void)configuration;
        return ObjectListImporter2280_2290::deserialize(inputStream, dataContainer, dataHeader);
    }

}; // ObjectListImporter2280

//==============================================================================

using ObjectListImporter2280 = Importer<ObjectList, DataTypeId::DataType_ObjectList2280>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
