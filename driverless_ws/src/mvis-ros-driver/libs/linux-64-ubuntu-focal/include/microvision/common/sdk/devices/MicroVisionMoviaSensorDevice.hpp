//==============================================================================
//! \file
//!
//! \brief MicroVision MOVIA B0 and L device support.
//!
//! \note Please note that using recent MOVIA sensors require the movia-device-plugin to be loaded!
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 21, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/io/UdpConfiguration.hpp>
#include <microvision/common/sdk/config/devices/MoviaSensorConfiguration.hpp>

#include <microvision/common/sdk/datablocks/ImporterBase.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2352.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>
#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/extension/DeviceFactory.hpp>
#include <microvision/common/sdk/extension/NetworkInterfaceFactory.hpp>
#include <microvision/common/sdk/io/iutp/NetworkDataPackageToIutpTranslator.hpp>
#include <microvision/common/sdk/io/intp/NetworkDataPackageToIntpTranslator.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/IntpToIdcPackageOfLdmiRawFrame2352Translator.hpp>
#include <microvision/common/sdk/io/icd/IcdToIdcPackageTranslator.hpp>
#include <microvision/common/sdk/io/icd/IutpToIcdPackageTranslator.hpp>
#include <microvision/common/sdk/io/icd/IcdToIdcPackageOfImage2404Translator.hpp>
#include <microvision/common/sdk/io/icd/IcdToIdcPackageOfZoneOccupationListA000Translator.hpp>
#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief \brief Helper function to create a MOVIA L industrial sensor device.
//! \param[in]  pointCloudUdpIpAddress  IP address for transmission of point cloud data. See manual on how to find this address. Adapt to your network setup.
//! \param[in]  pointCloudPort          Port for multicast and local IP address.
//! \returns  MOVIA device which is configured according to input parameters.
//!
//! \note You can still change configuration values manually (getDeviceConfiguration) after the device is created.
//! In case you want to choose the IP address of the network interface from which the point cloud data shall be received.
//! This is only relevant for applications which want to receive data from multiple MOVIA Lidar Sensors.
//! Linux: Set it to the multicast address.
//! Windows: Set it to the IP address of the ethernet adapter.
//!
//! \note This uses the MOVIA L sensor device from the SDK. If you need raw ldmi/aggregated frame and further configuration
//! options like scan configuration or image please use the "movia" device "L" after loading the movia-device-plugin.
//! See movia-device-plugin manual and MoviaConfigurationDemo.
//------------------------------------------------------------------------------
IdcDevicePtr createMoviaLDevice(const std::string& pointCloudUdpIpAddress, const uint16_t pointCloudPort = 12345U);

//==============================================================================
//! \brief Helper function to more easily create a MOVIA device
//! \param[in]  hwid                           The MOVIA sensor HWID. Printed on the sensor casing.
//! \param[in]  pointCloudUdpIpAddress         IP address with scope id part for transmission of point cloud data. See manual on how to find this address and the scope id. Adapt to your network setup.
//! \param[in]  controlRemoteIpAddress         Remote IP address for tcp control connection. Adapt to your VLAN network setup. Empty for default depending on HWID.
//! \param[in]  controlLocalInterfaceIpAddress Choose the local IP address of the network interface from which the control connection to the sensor shall be established. Adapt to your VLAN network setup. Empty for default.
//! \returns  MOVIA device which is configured according to input parameters.
//------------------------------------------------------------------------------
IdcDevicePtr createMoviaDeviceByHwid(const uint16_t hwid,
                                     const std::string& pointCloudUdpIpAddress,
                                     const std::string& controlRemoteIpAddress         = "",
                                     const std::string& controlLocalInterfaceIpAddress = "");

//==============================================================================
//! \brief Helper function to easily create a MOVIA device
//! \param[in]  deviceConfigDefaultParameterSet  The parameters which are used to configure the MOVIA device configuration. Change depending on sensor version and FOV.
//! \param[in]  scanConfigDefaultParameterSet    The parameters which are used to configure the MOVIA device scan configuration. Change depending on sensor version.
//! \param[in]  multicastAddress                 IP address where the udp pointcloud data from the sensor is received. Change address depending on sensor FOV and ScopeId depending on your network settings. See user manual.
//! \param[in]  pointCloudInterfaceIp            Choose the IP address of the network interface from which the point cloud data shall be received.
//!                                              This is only relevant for applications which want to receive data from multiple MOVIA Lidar Sensors.
//!                                              Linux: Set it to the multicast address.
//!                                              Windows: Set it to the IP address of the ethernet adapter.
//! \param[in]  controlInterfaceIp               Choose the IP address of the network interface from which the control connection to the sensor shall be established.
//! \param[in]  pointCloudPort                   Port for multicast and local IP address.
//! \param[in]  controlIpAddress                 Remote IP address for tcp control connection. Change address depending on sensor FOV or your VLAN setup!
//! \param[in]  controlPort                      Port for remote IP address.
//! \param[in]  controlInterfaceIp               Local interface IP address for tcp control connection. Change address depending on sensor FOV or your VLAN setup!
//! \param[in]  useAdditionalPointData           Use 2353 or 2354 ldmi raw data. Change depending on sensor. Set by defaultparameterset if without value.
//! \param[in]  useE2E                           Use end to end data protection for ldmi static message. Change depending on sensor. Set by defaultparameterset if without value.
//! \param[in]  useControlConnection             Use TCP SOME/IP control connection. False for MOVIA L.
//! \returns  Configured MOVIA device.
//------------------------------------------------------------------------------
IdcDevicePtr createMoviaDevice(const std::string& deviceConfigDefaultParameterSet,
                               const std::string& scanConfigDefaultParameterSet,
                               const std::string& multicastAddress,
                               const std::string& pointCloudInterfaceIp,
                               uint16_t pointCloudPort,
                               const std::string& controlIpAddress,
                               uint16_t controlPort,
                               const std::string& controlInterfaceIp,
                               Optional<bool> useAdditionalPointData = nullopt,
                               Optional<bool> useE2E                 = nullopt,
                               Optional<bool> useControlConnection   = nullopt);

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Access MOVIA Lidar Sensor device.
//!
//! \note This does not support all MOVIA sensors. Use movia-device-plugin when in doubt.
//!
//! The configuration of the MOVIA Lidar Sensor device is basically the default udp configuration.
//! So every device receives only data from one network source.
//!
//! \extends IdcDevice
//------------------------------------------------------------------------------
class MicroVisionMoviaSensorDevice final : public IdcDevice
{
public:
    //========================================
    //! \brief Type name of MOVIA device.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::MicroVisionMoviaSensorDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MicroVisionMoviaSensorDevice();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~MicroVisionMoviaSensorDevice() override;

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

public: // implements Device
    //========================================
    //! \brief Get the type name of this device.
    //! \returns The type name of this device.
    //----------------------------------------
    const std::string& getDeviceType() const override;

    //========================================
    //! \brief Get the last error which has occurred.
    //! \returns The exception of the last occurred error.
    //----------------------------------------
    std::exception_ptr getLastError() const override;

    //========================================
    //! \brief Checks if a connection has been established.
    //! \returns Either \c true if connection is established or otherwise \c false.
    //----------------------------------------
    bool isConnected() const override;

    //========================================
    //! \brief Checks that a connection is established and packages are still processed.
    //! \returns Either \c true if connection is established or work is ongoing on received data or otherwise \c false.
    //----------------------------------------
    bool isWorking() override;

    //========================================
    //! \brief Connects the device to the physical device via network.
    //----------------------------------------
    void connect() override;

    //========================================
    //! \brief Disconnects the device from the physical device.
    //----------------------------------------
    void disconnect() override;

    //========================================
    //! \brief Get the current configuration of the device.
    //! \returns The current configuration.
    //!
    //! \sa UdpConfiguration
    //----------------------------------------
    ConfigurationPtr getDeviceConfiguration() const override;

    //========================================
    //! \brief Set the configuration of this device.
    //! \param[in] deviceConfiguration  The shared pointer to new configuration of this device.
    //! \return Either \c true if this device can be configured with those configuration, otherwise \c false.
    //!
    //! \sa UdpConfiguration
    //----------------------------------------
    bool setDeviceConfiguration(const ConfigurationPtr& deviceConfiguration) override;

    //========================================
    //! \brief Lock current configuration for use.
    //!
    //! The locked configuration will be used for all received data packages until unlock.
    //! Configuration will be locked during connection - some properties may still be updated when set
    //! depending on device implementation.
    //! For offline use with processDataPackage the configuration can be manually unlocked to change
    //! configuration properties. Then before the next processDataPackage it has to be locked again.
    //!
    //! \return Either \c true if the configuration is valid, otherwise \c false.
    //----------------------------------------
    bool lockConfiguration() override;

    //========================================
    //! \brief Released locked configuration.
    //!
    //! This does not work while the device is connected.
    //!
    //! \return Either \c true if the configuration is unlocked, otherwise \c false.
    //----------------------------------------
    bool unlockConfiguration() override;

    //========================================
    //! \brief Process a data package as sensor input from another source (e.g. as read from a file).
    //! \param[in] dataPackage  The data package to process.
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    bool processDataPackage(const DataPackagePtr& dataPackage) override;

private:
    //========================================
    //! \brief Main method of thread which observes the receiver.
    //! \param[in] worker  Excecuting Background worker m_observerWorker.
    //! \returns Either \c true too keep thread alive or otherwise \c false.
    //----------------------------------------
    bool observerMain(BackgroundWorker& worker);

    //========================================
    //! \brief Publish idc package as received by device to listeners.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    void publishIdcPackage(const IdcDataPackagePtr& data);

    //========================================
    //! \brief Method to be called if a new IdcDataPackage has been received.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    void notifyListeners(const IdcDataPackagePtr& data);

private:
    //========================================
    //! \brief Increment index of received packages.
    //----------------------------------------
    ThreadSafe<int64_t> m_packageIndex;

    //========================================
    //! \brief The Uri which denotes the packets dedicated for this device.
    //----------------------------------------
    Uri m_packageUri;

    //========================================
    //! \brief Receiver observer thread.
    //----------------------------------------
    BackgroundWorker m_observerWorker;

    //========================================
    //! \brief Receiver for UDP data packages.
    //----------------------------------------
    NetworkInterfaceUPtr m_udpReceiver;

    //========================================
    //! \brief Current configuration.
    //----------------------------------------
    MoviaTransportProtocol m_sensorTransportProtocol;

    //========================================
    //! \brief Network to iutp data package translator.
    //----------------------------------------
    NetworkDataPackageToIutpTranslator m_iutpPackageForIntpTranslator;

    //========================================
    //! \brief Iutp to intp data package translator.
    //----------------------------------------
    NetworkDataPackageToIntpTranslator m_intpPackageForIdcTranslator;

    //========================================
    //! \brief Data per sensor received.
    //----------------------------------------
    IntpToIdcPackageOfLdmiRawFrame2352Translator m_idcPackageForDeviceTranslator;

    //========================================
    //! \brief Network to iutp data package translator.
    //!
    //! \note Used when receiving ldmi raw is disabled. Then icd data will be received (already processed on the MOVIA).
    //! For all successfully translated packages the icd translator is called.
    //----------------------------------------
    common::sdk::NetworkDataPackageToIutpTranslator m_iutpPackageForIcdTranslator;

    //========================================
    //! \brief Iutp to icd data package translator.
    //!
    //! \note Used when receiving ldmi raw is disabled. Then icd data will be received (already processed on the MOVIA).
    //! For all successfully translated packages the idc translator is called.
    //----------------------------------------
    common::sdk::IutpToIcdPackageTranslator m_icdPackageForIdcTranslator;

    //========================================
    //! \brief Icd data to idc package translator.
    //!
    //! \note Used when receiving ldmi raw is disabled. Then icd data will be received (already processed on the MOVIA).
    //! All successfully translated packages are published to the device.
    //----------------------------------------
    common::sdk::IcdToIdcPackageTranslator m_idcPackageForDeviceTranslatorWithMpl;

    //========================================
    //! \brief Icd data to idc image package translator.
    //!
    //! \note Used when receiving ldmi raw is disabled. Then icd data will be received (already processed on the MOVIA).
    //! All successfully translated packages are published to the device.
    //----------------------------------------
    IcdToIdcPackageOfImage2404Translator m_idcPackageForDeviceTranslatorWithImage;

    //========================================
    //! \brief Icd data to idc zone occupation list package translator.
    //!
    //! \note Used when receiving ldmi raw is disabled. Then icd data will be received (already processed on the MOVIA).
    //! All successfully translated packages are published to the device.
    //----------------------------------------
    IcdToIdcPackageOfZoneOccupationListA000Translator m_idcPackageForDeviceTranslatorWithZoneOccupationList;

    //========================================
    //! \brief Current configuration.
    //----------------------------------------
    ConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked udp configuration.
    //----------------------------------------
    UdpConfigurationPtr m_lockedConfiguration;
}; // class MicroVisionMoviaSensorDevice

//==============================================================================
//! \brief Pointer type of MOVIA Lidar Sensor device.
//------------------------------------------------------------------------------
using MicroVisionMoviaSensorDevicePtr = std::unique_ptr<MicroVisionMoviaSensorDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
