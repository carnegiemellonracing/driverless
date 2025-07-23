//==============================================================================
//! \file
//!
//! \brief MVIS ECU device implementation.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 02, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/MvisEcuConfiguration.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>

#include <microvision/common/sdk/io/idc/DataPackageToIdcTranslator.hpp>

#include <microvision/common/sdk/commands/CommandInterface.hpp>
#include <microvision/common/sdk/commands/legacy/ecu/AppbaseEcuCommand.hpp>

#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/io/NetworkInterface.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/logging/logging.hpp>

#include <future>
#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief MVIS ECU device implementation.
//!
//! \extends IdcDevice
//! \extends CommandInterface
//------------------------------------------------------------------------------
class MvisEcuDevice final : public IdcDevice, public CommandInterface
{
public:
    //========================================
    //! \brief Type name of this device.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //========================================
    //! \brief Default tcp port.
    //----------------------------------------
    static constexpr MICROVISION_SDK_API uint16_t defaultTcpPort{12002U};

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::MvisEcuDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MvisEcuDevice();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~MvisEcuDevice() override;

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
    //! \sa MvisEcuConfiguration
    //----------------------------------------
    ConfigurationPtr getDeviceConfiguration() const override;

    //========================================
    //! \brief Set the configuration of this device.
    //! \param[in] deviceConfiguration  The shared pointer to new configuration of this device.
    //! \return Either \c true if this device can be configured with those configuration, otherwise \c false.
    //!
    //! \sa MvisEcuConfiguration
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

public: // implements CommandInterface
    //========================================
    //! \brief Get available request types for MVIS Appbase ECU command.
    //! \returns Available request types.
    //--------------------------------------
    const std::vector<std::string> getAvailableRequestTypes() const override;

private:
    //========================================
    //! \brief Main method of thread which observes the receiver.
    //! \param[in] worker  Executing Background worker m_observerWorker.
    //! \returns Either \c true too keep thread alive or otherwise \c false.
    //----------------------------------------
    bool observerMain(BackgroundWorker& worker);

    //========================================
    //! \brief Publish idc package as received by device to listeners.
    //! \param[in] data  Idc data package
    //----------------------------------------
    void publishIdcPackage(const IdcDataPackagePtr& data);

    //========================================
    //! \brief Notify listeners with idc data package.
    //! \param[in] data  Idc data package.
    //----------------------------------------
    void notifyListeners(const IdcDataPackagePtr& data);

private:
    //========================================
    //! \brief Callback function to send 'set filter' command when the 'data type range vector' property has changed.
    //!
    //! \param[in] property    The 'data type range vector' property
    //!
    //! \note Will override locked configuration property range vector.
    //----------------------------------------
    void onConfigurationPropertyRangeVectorChanged(const ConfigurationProperty& property, const Any&);

    //========================================
    //! \brief Create command for MVIS Appbase ECU.
    //! \param[in] Request  type of the command.
    //! \returns The freshly created command.
    //--------------------------------------
    CommandPtr createCommandInternal(const std::string& requestType) override;

    //========================================
    //! \brief Send command by parameters or next queue entry.
    //! \param[in] parameters  Command parameters to send or queue.
    //--------------------------------------
    void handleCommandExecution(const AppbaseEcuCommand::SendFunctionParametersPtr& parameters);

private:
    //========================================
    //! \brief The Uri which denotes the packets dedicated for this device.
    //----------------------------------------
    Uri m_packageUri;

    //========================================
    //! \brief  Increment index of received packages.
    //----------------------------------------
    ThreadSafe<int64_t> m_packageIndex;

    //========================================
    //! \brief Reassembles data packages sent to this device to regular idc data packages.
    //----------------------------------------
    DataPackageToIdcTranslator m_idcPackageForDeviceTranslator;

    //========================================
    //! \brief Current configuration.
    //----------------------------------------
    MvisEcuConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked configuration.
    //----------------------------------------
    MvisEcuConfigurationPtr m_lockedConfiguration;

    //========================================
    //! \brief TCP network interface to connect against ECU.
    //----------------------------------------
    NetworkInterfaceUPtr m_tcpConnectionInterface;

    //========================================
    //! \brief Thread to observe input queue of tcp client.
    //----------------------------------------
    BackgroundWorker m_observerWorker;

    //========================================
    //! \brief Queue of command data which has to be send.
    //----------------------------------------
    std::list<AppbaseEcuCommand::SendFunctionParametersPtr> m_commandQueue;

    //========================================
    //! \brief Current command data which is sended.
    //----------------------------------------
    AppbaseEcuCommand::SendFunctionParametersPtr m_commandInProcess;

    //========================================
    //! \brief Command 'set range filter' to send.
    //----------------------------------------
    CommandPtr m_setRangeFilterCommand;
}; // class MvisEcuDevice

//==============================================================================
//! \brief Pointer type of Ecu device.
//------------------------------------------------------------------------------
using MvisEcuDevicePtr = std::unique_ptr<MvisEcuDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
