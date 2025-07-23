//==============================================================================
//! \file
//!
//! \brief MVIS Perception Development Datasource (LDE/PDD) device configuration.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 02, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/DeviceConfiguration.hpp>
#include <microvision/common/sdk/config/io/UdpConfiguration.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>

#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerHeader.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Configuration for the MVIS LDE/PDD device.
//!
//! Example lde/pdd device configuration:
//! \code
//! auto deviceConfig = ConfigurationFactory::getInstance().createConfiguration(MvisLdeConfiguration::typeName);
//! deviceConfig->trySetValue("multicast_ip", makeIp("239.1.2.5")); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("local_port", uint16_t{12349}); // if false: configuration property does not exists or type is incompatible!
//!
//! const MvisLdeConfiguration::DeviceIdMap idMapping{{173457843, 0}, {245726774, 1}};
//! deviceConfig->trySetValue("device_mapping", idMapping); // if false: configuration property does not exist or type is incompatible!
//! deviceConfig->trySetValue("icd_output", false); // if return value is false: configuration property does not exist or type is incompatible!
//! deviceConfig->trySetValue("all_custom_output", false); // if return value is false: configuration property does not exist or type is incompatible!
//! deviceConfig->trySetValue("use_receiving_time", false); // if return value is false: configuration property does not exist or type is incompatible!
//! deviceConfig->trySetValue("use_gdtp_timestamp", false); // if return value is false: configuration property does not exist or type is incompatible!
//!
//! device->setDeviceConfiguration(deviceConfig); // if false: device configuration failed
//! \endcode
//!
//! New configuration properties added:
//! Property Name      | Type        | Description                                                                               | Default
//! ------------------ | ----------- | ----------------------------------------------------------------------------------------- | -------------
//! device_mapping     | DeviceIdMap | Global device to device setup ids                                                         | {}
//! scan_output        | bool        | Enable scan data container output for point cloud                                         | true
//! icd_output         | bool        | Enable icd custom data container output for all icd data                                  | true
//! all_custom_output  | bool        | Enable custom data container output for all data                                          | false
//! use_receiving_time | bool        | Switch between meta information from data and receiving time                              | false
//! use_gdtp_timestamp | bool        | Replace idc header time with GDTP header timestamp entry (overrides 'use_receiving_time') | false
//!
//! \sa UdpConfiguration
//------------------------------------------------------------------------------
class MvisLdeConfiguration : public UdpConfiguration
{
public:
    //========================================
    //! \brief Configuration type name
    //----------------------------------------
    static MICROVISION_SDK_API const std::string typeName;

    //========================================
    //! \brief Old configuration type name LDE.
    //!
    //! \deprecated Use pdd.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string typeNameLDE;

    //========================================
    //! \brief Unique config id for property 'device_mapping'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string deviceMappingConfigId;

    //========================================
    //! \brief Unique config id for property 'scan_output'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string scanOutputConfigId;

    //========================================
    //! \brief Unique config id for property 'icd_output'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string icdOutputConfigId;

    //========================================
    //! \brief Unique config id for property 'all_custom_output'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string allCustomOutputConfigId;

    //========================================
    //! \brief Unique config id for property 'use_receiving_time'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string useReceivingTimeConfigId;

    //========================================
    //! \brief Unique config id for property 'use_gdtp_time'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string useGdtpTimeConfigId;

public:
    //========================================
    //!\brief A mapping of uint64 to uint8.
    //----------------------------------------
    using DeviceIdMap = std::map<uint64_t, uint8_t>;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    MvisLdeConfiguration();

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    MvisLdeConfiguration(const MvisLdeConfiguration& other);

    //========================================
    //! \brief Disable move constructor to ensure thread safety
    //----------------------------------------
    MvisLdeConfiguration(UdpConfiguration&&) = delete;

    //========================================
    //! \brief Destructor
    //----------------------------------------
    ~MvisLdeConfiguration() override = default;

public:
    //========================================
    //! \brief Return the configuration type
    //! \returns Configuration type
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get copy of configuration
    //! \returns Pointer to newly copied configuration
    //----------------------------------------
    ConfigurationPtr copy() const override;

public:
    //========================================
    //! \brief Get the device id mapping from global device id to setup device.
    //! \returns Mapping of global device ids to device setup id.
    //----------------------------------------
    ConfigurationPropertyOfType<DeviceIdMap>& getDeviceMapping();

    //========================================
    //! \brief Get the scan output flag configuration property to setup device.
    //!
    //! This flag indicates the device puts out received point clouds as MVIS SDK scan data container.
    //! \returns Bool output flag configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<bool>& getScanOutput();

    //========================================
    //! \brief Get the icd output flag configuration property to setup device.
    //!
    //! This flag indicates the device puts out received icd data as custom data container containing the icd data.
    //!
    //! \note Attention: Invalid icd data can be passed through if used when receiving non icd types!
    //!
    //! \returns Bool output flag configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<bool>& getIcdOutput();

    //========================================
    //! \brief Get the all data as custom data container output flag configuration property to setup device.
    //!
    //! This flag indicates the device puts out all received data as custom data container containing the data as content.
    //! \returns Bool output flag configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<bool>& getAllCustomOutput();

    //========================================
    //! \brief Get the use receiving time flag configuration property to setup device.
    //!
    //! This flag indicates the device uses meta information from data to set ntp time of idc data header for the data
    //! container or it uses the measured receiving time.
    //! \returns Bool output flag configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<bool>& getUseReceivingTime();

    //========================================
    //! \brief Get the use gdtp header time flag configuration property to setup device.
    //!
    //! This flag indicates the device uses the gdtp header timestamp entry to set ntp time of idc data header for the data
    //! container.
    //! \returns Bool output flag configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<bool>& getUseGdtpTime();

private:
    //========================================
    //! \brief Property for mapping of global device ids to device setup ids.
    //----------------------------------------
    ConfigurationPropertyOfType<DeviceIdMap> m_deviceIds;

    //========================================
    //! \brief Property for the MVIS SDK scan output flag configuration.
    //----------------------------------------
    ConfigurationPropertyOfType<bool> m_scanOutput;

    //========================================
    //! \brief Property for the custom icd output flag configuration.
    //! \note Attention: The PerceptionDataInfo deserialization is unable to securely detect whether a data type is icd or not.
    //! Invalid icd data may be wrapped into a custom data container if used when receiving non icd types!
    //----------------------------------------
    ConfigurationPropertyOfType<bool> m_icdOutput;

    //========================================
    //! \brief Property for the all custom output flag configuration.
    //----------------------------------------
    ConfigurationPropertyOfType<bool> m_allCustomOutput;

    //========================================
    //! \brief Property for the use receiving time flag configuration.
    //----------------------------------------
    ConfigurationPropertyOfType<bool> m_useReceivingTime;

    //========================================
    //! \brief Property for the use gdtp header timestamp entry flag configuration.
    //----------------------------------------
    ConfigurationPropertyOfType<bool> m_useGdtpTime;
};

//=================================================
//! \brief Nullable MvisLdeConfiguration pointer
//-------------------------------------------------
using MvisLdeConfigurationPtr = std::shared_ptr<MvisLdeConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
