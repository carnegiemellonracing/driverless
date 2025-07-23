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
//! \date Jan 17, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2291.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<ObjectList2291, DataTypeId::DataType_ObjectList2291>
  : public RegisteredImporter<ObjectList2291, DataTypeId::DataType_ObjectList2291>
{
public:
    //========================================
    //! \brief Empty constructor calling base.
    //----------------------------------------
    Importer() : RegisteredImporter() {}

    //========================================
    //! \brief Copy construction is forbidden.
    //----------------------------------------
    Importer(const Importer&) = delete;

    //========================================
    //! \brief Assignment construction is forbidden.
    //----------------------------------------
    Importer& operator=(const Importer&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Importer() override = default;

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

}; // ObjectList2291Importer2291

//==============================================================================

using ObjectList2291Importer2291 = Importer<ObjectList2291, DataTypeId::DataType_ObjectList2291>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
