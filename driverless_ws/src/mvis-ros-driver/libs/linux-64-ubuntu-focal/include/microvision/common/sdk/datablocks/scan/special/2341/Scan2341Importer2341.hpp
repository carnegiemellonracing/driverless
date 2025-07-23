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
//! \date Jun 24, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2341/Scan2341.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to import a processed low bandwidth MOVIA scan from a binary
//!        idc data block to deserialize it into a scan data container.
//------------------------------------------------------------------------------
template<>
class MICROVISION_SDK_DEPRECATED Importer<Scan2341, DataTypeId::DataType_Scan2341>
  : public RegisteredImporter<Scan2341, DataTypeId::DataType_Scan2341>
{
public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    Importer() : RegisteredImporter() {}

    //========================================
    //! Copy construction is forbidden.
    //----------------------------------------
    Importer(const Importer&) = delete;

    //========================================
    //! Assignment construction is forbidden.
    //----------------------------------------
    Importer& operator=(const Importer&) = delete;

public: // implements ImporterBase
    //=================================================
    //! \brief Get the size in bytes that the object occupies in serialized binary form.
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
    //!\brief Convert from byte stream (deserialization).
    //!\param[in]      is  Input data stream
    //!\param[in, out] c   Data container.
    //!\return \c True if deserialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool deserialize(std::istream& is, ScannerDirectionListIn2341& c);

    //========================================
    //!\brief Convert from byte stream (deserialization).
    //!\param[in]      is  Input data stream
    //!\param[in, out] c   Data container.
    //!\return \c True if deserialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool deserialize(std::istream& is, ScannerInfoIn2341& c);

    //========================================
    //!\brief Convert from byte stream (deserialization).
    //!\param[in]      is  Input data stream
    //!\param[in, out] c   Data container.
    //!\return \c True if deserialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool deserialize(std::istream& is, ScanPointIn2341& c);

    //========================================
    //!\brief Convert from byte stream (deserialization).
    //!\param[in]      is  Input data stream
    //!\param[in, out] c   Data container.
    //!\return \c True if deserialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool deserialize(std::istream& is, ScanPointRowIn2341& c);

}; // Scan2341Importer2341

//==============================================================================

using Scan2341Importer2341 = Importer<microvision::common::sdk::Scan2341, DataTypeId::DataType_Scan2341>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
