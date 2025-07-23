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
//! \date Mar 15, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/eventtag/special/UserEventTag7010.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<UserEventTag7010, DataTypeId::DataType_UserEventTag7010>
  : virtual public RegisteredImporter<UserEventTag7010, DataTypeId::DataType_UserEventTag7010>
{
public:
    static constexpr std::streamsize serializedBaseSize{45 + UserEventTag7010::nbOfReserved};

public:
    Importer() : RegisteredImporter<UserEventTag7010, DataTypeId::DataType_UserEventTag7010>() {}
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

}; //UserEventTag7010Importer7010

//==============================================================================

using UserEventTag7010Importer7010 = Importer<UserEventTag7010, DataTypeId::DataType_UserEventTag7010>;

//==============================================================================

template<>
void readBE<UserEventTag7010::TagClass>(std::istream& is, UserEventTag7010::TagClass& tc);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
