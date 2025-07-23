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
//! \date Jan 18, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to import a processed MOVIA scan from a binary idc data block to deserialize
//!        it into a Scan2342 data container.
//------------------------------------------------------------------------------
template<>
class Importer<Scan2342, DataTypeId::DataType_Scan2340>
  : public RegisteredImporter<Scan2342, DataTypeId::DataType_Scan2340>
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
    //!
    //! \return  The number of bytes used for serialization.
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
    //!
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //-------------------------------------------------
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;

private:
    static constexpr const char* loggerId{"microvision::common::sdk::Scan2342Importer2340"};
    static microvision::common::logging::LoggerSPtr logger;

}; //Scan2342Importer2340

//==============================================================================

using Scan2342Importer2340 = Importer<microvision::common::sdk::Scan2342, DataTypeId::DataType_Scan2340>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
