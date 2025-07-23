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
//! \date Dec 19, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/CarriageWayList.hpp>
#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<CarriageWayList, DataTypeId::DataType_CarriageWayList6972>
  : public RegisteredImporter<CarriageWayList, DataTypeId::DataType_CarriageWayList6972>
{
public:
    Importer()                = default;
    Importer(const Importer&) = delete;
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

}; //CarriageWayListImporter6972

//==============================================================================

using CarriageWayListImporter6972 = Importer<CarriageWayList, DataTypeId::DataType_CarriageWayList6972>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
