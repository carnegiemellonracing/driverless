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
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/marker/special/MarkerList6820.hpp>
#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Special importer (0x6820) for MarkerList6820.
//!
//! This importer provides the means to import a MarkerList6820 from a 0x6820
//! serialization.
//------------------------------------------------------------------------------
template<>
class Importer<MarkerList6820, DataTypeId::DataType_Marker6820>
  : public RegisteredImporter<MarkerList6820, DataTypeId::DataType_Marker6820>
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

private:
    //========================================
    //! \brief Deserialize a string \a targetString from \a is.
    //!
    //! The length of the string is serialized before the string content as
    //! a uint16_t.
    //!
    //! \param[in,out] is            Input stream, the length of the following string
    //!                              and the string itself is read from.
    //! \param[out]    targetString  The string, where the result of the
    //!                              deserialization is stored.
    //----------------------------------------
    static void deserializeString(std::istream& is, std::string& targetString);
}; //LogPolygonList2dImporter6817

//==============================================================================

using MarkerList6820Importer6820 = Importer<MarkerList6820, DataTypeId::DataType_Marker6820>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
