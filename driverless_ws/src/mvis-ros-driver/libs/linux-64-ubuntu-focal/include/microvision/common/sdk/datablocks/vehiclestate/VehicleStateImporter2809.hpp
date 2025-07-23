//==============================================================================
//! \file
//!
//! \brief Imports vehicleState type 0x2809 to general vehicle state container
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 24, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleState.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<VehicleState, DataTypeId::DataType_VehicleStateBasic2809>
  : public RegisteredImporter<VehicleState, DataTypeId::DataType_VehicleStateBasic2809>
{
public:
    //========================================
    //! \brief Constructor.
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

}; // VehicleStateImporter2809

//==============================================================================

using VehicleStateImporter2809 = Importer<VehicleState, DataTypeId::DataType_VehicleStateBasic2809>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
