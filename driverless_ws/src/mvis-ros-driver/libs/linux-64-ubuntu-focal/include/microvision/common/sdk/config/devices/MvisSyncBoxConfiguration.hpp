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
//! \date Jan 30, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/DeviceConfiguration.hpp>
#include <microvision/common/sdk/config/io/UdpConfiguration.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Abstract base for SyncBox device configurations.
//!
//! \extends microvision::common::sdk::DeviceConfiguration
//------------------------------------------------------------------------------
class MvisSyncBoxConfiguration : public DeviceConfiguration, public UdpConfiguration
{
public:
    //========================================
    //! \brief Configuration type name.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //========================================
    //! \brief The default UDP port where the sync box device is sending packets to.
    //----------------------------------------
    static constexpr MICROVISION_SDK_API uint16_t defaultUdpPort{6320U};

public:
    //========================================
    //! \brief Construct and update SyncBox properties.
    //----------------------------------------
    MvisSyncBoxConfiguration();

    //========================================
    //! \brief Copy constructor to copy and update SyncBox properties.
    //! \param[in] other  Other MvisSyncBoxConfiguration to copy.
    //----------------------------------------
    MvisSyncBoxConfiguration(const MvisSyncBoxConfiguration& other);

    //========================================
    //! \brief Disabled move constructor because of thread safe guarantee.
    //----------------------------------------
    MvisSyncBoxConfiguration(MvisSyncBoxConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~MvisSyncBoxConfiguration() override;

public: // implements Configuration
    //========================================
    //! \brief Get type of configuration to match with.
    //! \returns Configuration type.
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get copy of configuration.
    //! \returns Pointer to new copied Configuration.
    //----------------------------------------
    ConfigurationPtr copy() const override;

}; // class MvisSyncBoxConfiguration

//==============================================================================
//! \brief Nullable MvisSyncBoxConfiguration pointer.
//------------------------------------------------------------------------------
using SyncBoxConfigurationPtr = std::shared_ptr<MvisSyncBoxConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
