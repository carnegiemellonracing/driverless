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
//! \date Feb 21, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/ConfigurationPropertyOfType.hpp>
#include <microvision/common/sdk/config/Configuration.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Abstract base for extra port configurations.
//!
//! \extends microvision::common::sdk::Configuration
//------------------------------------------------------------------------------
class ExtraPortConfiguration : public virtual Configuration
{
public:
    //==============================================================================
    //! \brief Unique config id for property of 'local gps port'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string localGpsPortConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'remote gps port'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string remoteGpsPortConfigId;

public:
    //========================================
    //! \brief Construct and update extra port properties with optional default values.
    //! \param[in] defaultLocalGpsPort      (Optional) Default value for local gps port, default is \c 0.
    //! \param[in] defaultRemoteGpsPort     (Optional) Default value for remote gps port, default is \c 0.
    //----------------------------------------
    ExtraPortConfiguration(const uint16_t defaultLocalGpsPort = 0U, const uint16_t defaultRemoteGpsPort = 0U);

    //========================================
    //! \brief Copy constructor to copy and update extra port properties.
    //! \param[in] other  Other ExtraPortConfiguration to copy.
    //----------------------------------------
    ExtraPortConfiguration(const ExtraPortConfiguration& other);

    //========================================
    //! \brief Disabled move constructor to ensure thread-safety.
    //----------------------------------------
    ExtraPortConfiguration(ExtraPortConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ExtraPortConfiguration() override;

public:
    //========================================
    //! \brief Get local gps port configuration property.
    //!
    //! To address the endpoint on the local maschine.
    //!
    //! \note Will used mostly for udp network connections.
    //! \returns Local gps port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t>& getLocalGpsPort();

    //========================================
    //! \brief Get remote gps port configuration property.
    //!
    //! To address the endpoint on the remote maschine.
    //!
    //! \note Will used mostly for tcp network connections.
    //! \returns Remote gps port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t>& getRemoteGpsPort();

protected:
    //========================================
    //! \brief Local gps port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t> m_localGpsPort;

    //========================================
    //! \brief Remote gps port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t> m_remoteGpsPort;

}; // class ExtraPortConfiguration

//==============================================================================
//! \brief Nullable ExtraPortConfiguration pointer.
//------------------------------------------------------------------------------
using ExtraPortConfigurationPtr = std::shared_ptr<ExtraPortConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
