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
//! \date Jun 12, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcDevice.hpp>
#include <microvision/common/sdk/extension/DeviceFactoryExtension.hpp>
#include <microvision/common/sdk/extension/Extendable.hpp>

#include <microvision/common/logging/logging.hpp>

#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Factory extension to create devices.
//!
//! This singleton factory is extendable and will create devices.
//!
//! \extends Extendable<DeviceFactoryExtension>
//------------------------------------------------------------------------------
class DeviceFactory final : public Extendable<DeviceFactoryExtension>
{
private:
    //========================================
    //! \brief Logger name for setup logger configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::DeviceFactory";

    //========================================
    //! \brief Provides common logger interface.
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Get the singleton instance.
    //! \return Static instance of DeviceFactory.
    //----------------------------------------
    static DeviceFactory& getInstance();

private:
    //========================================
    //! Constructor registering all MVIS SDK device extensions.
    //!
    //! Devices which are not delivered as plugins to the customer have to be registered manually.
    //!
    //! \note When adding new devices with the sdk they have to be registered here.
    //----------------------------------------
    DeviceFactory();

public:
    //========================================
    //! \brief Copy constructor removed.
    //!
    //! No copies of the singleton.
    //----------------------------------------
    DeviceFactory(DeviceFactory const&) = delete;

    //========================================
    //! \brief Assignment copy construction removed.
    //!
    //! No copies of the singleton.
    //----------------------------------------
    void operator=(DeviceFactory const&) = delete;

public:
    //========================================
    //! \brief Get all registered device types.
    //! \return Registered device types which can be created by this factory.
    //----------------------------------------
    const std::list<std::string>& getAllRegisteredDeviceTypes() const;

    //========================================
    //! \brief Create device from type name.
    //! \param[in] typeName         String which identifies the device type.
    //! \param[in] configuration    Configuration to set on device.
    //! \return Device created by the factory or \c nullptr.
    //----------------------------------------
    DevicePtr createDeviceFromTypeName(const std::string& typeName,
                                       const ConfigurationPtr& configuration = nullptr) const;

    //========================================
    //! \brief Create device from type name as IdcDevice.
    //!
    //! The created device which implements IdcDevice provides support
    //! of data container and idc data packages listeners and more.
    //!
    //! \note If the device does not implement IdcDevice it will return nullptr.
    //! \param[in] typeName         String which identifies the device type.
    //! \param[in] configuration    Configuration to set on device.
    //! \return IdcDevice created by the factory or \c nullptr.
    //----------------------------------------
    IdcDevicePtr createIdcDeviceFromTypeName(const std::string& typeName,
                                             const ConfigurationPtr& configuration = nullptr) const;

public:
    //========================================
    //! \brief Register this device factory for example from a plugin.
    //! \param[in] ext  The device factory extension.
    //! \return The device factory extension registered.
    //----------------------------------------
    const std::shared_ptr<DeviceFactoryExtension>
    registerExtension(const std::shared_ptr<DeviceFactoryExtension>& ext) override;

    //========================================
    //! \brief Remove this device factory extension.
    //! \param[in] ext  The device factory extension.
    //----------------------------------------
    void unregisterExtension(const std::shared_ptr<DeviceFactoryExtension>& ext) override;

private:
    //========================================
    //! \brief Type names of registered device types.
    //----------------------------------------
    std::list<std::string> m_registeredTypeNames;

}; // class DeviceFactory

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
