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
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340.hpp>
#include <microvision/common/sdk/TransformationMatrix3d.hpp>
#include <microvision/common/sdk/Vector3.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to import a processed low bandwidth MOVIA scan from a binary idc data block to deserialize
//!        it into a scan2340 data container.
//------------------------------------------------------------------------------
template<>
class Importer<Scan2340, DataTypeId::DataType_Scan2342>
  : public RegisteredImporter<Scan2340, DataTypeId::DataType_Scan2342>
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

public: // implements Configurable
    //========================================
    //! \brief Get supported types of configuration.
    //!
    //! Supported means that this device can interpret imported data packages using this configuration.
    //! Configuration type is a human readable unique string name of the configuration
    //! used to address it in code.
    //!
    //! \returns All supported configuration types.
    //----------------------------------------
    const std::vector<std::string>& getConfigurationTypes() const override;

    //========================================
    //! \brief Get whether a configuration is mandatory for this Configurable.
    //!
    //! For Scan2340Importer2342 this is always false.
    //!
    //! \return \c true if a configuration is mandatory for this Configurable,
    //!         \c false otherwise.
    //----------------------------------------
    bool isConfigurationMandatory() const override;

private:
    //========================================
    //! \brief Convert the scan from the imported Scan2342 into a
    //!        Scan2340.
    //!
    //! \param[out] container  The Scan2340 to be filled.
    //! \param[in]  scan2342   The imported Scan2342 to be converted into
    //!                        the Scan2340.
    //! \param[in]  skipInvalidPoints
    //!                        If \c true the resulting Scan2340 points
    //!                        array will not contain any invalid points,
    //!                        i.e. with radial distance 0.
    //!                        If \c false, all points from \a scan2342 will
    //!                        be copied regardless whether invalid or not.
    //----------------------------------------
    bool convertScan(microvision::common::sdk::Scan2340* const container,
                     const Scan2342& scan2342,
                     const bool skipInvalidPoints) const;

    //========================================
    //! \brief Convert the scan info from the imported Scan2342 into a
    //!        Scan2340 scanner info.
    //!
    //! \param[out] container  The Scan2340 to be filled.
    //! \param[in]  scan2342   The imported Scan2342 to be converted into
    //!                        the Scan2340.
    //----------------------------------------
    void convertScannerInfo(microvision::common::sdk::Scan2340* const container, const Scan2342& scan2342) const;

    //========================================
    //! \brief Convert the scan points from the imported Scan2342 into a
    //!        Scan2340 scan point.
    //!
    //! \param[out] container  The Scan2340 to be filled.
    //! \param[in]  scan2342   The imported Scan2342 to be converted into
    //!                        the Scan2340.
    //! \param[in] skipInvalidPoints
    //!                        If \c true the resulting Scan2340 points
    //!                        array will not contain any invalid points,
    //!                        i.e. with radial distance 0.
    //!                        If \c false, all points from \a scan2342 will
    //!                        be copied regardless whether invalid or not.
    //----------------------------------------
    void convertScanPoints(microvision::common::sdk::Scan2340* const container,
                           const Scan2342& scan2342,
                           const bool skipInvalidPoints) const;

private:
    static constexpr const char* loggerId{"microvision::common::sdk::Scan2340Importer2342"};
    static microvision::common::logging::LoggerSPtr logger;

}; //Scan2340Importer2342

//==============================================================================

using Scan2340Importer2342 = Importer<microvision::common::sdk::Scan2340, DataTypeId::DataType_Scan2342>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
