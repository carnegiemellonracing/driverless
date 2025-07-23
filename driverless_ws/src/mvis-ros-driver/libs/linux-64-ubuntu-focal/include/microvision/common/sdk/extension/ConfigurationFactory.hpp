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
//! \date Feb 04, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/ConfigurationFactoryExtension.hpp>
#include <microvision/common/sdk/extension/Extendable.hpp>

#include <microvision/common/logging/logging.hpp>

#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Factory to create configurations.
//!
//! This singleton factory is extendable and will create configurations.
//------------------------------------------------------------------------------
class ConfigurationFactory final : public Extendable<ConfigurationFactoryExtension>
{
private:
    //========================================
    //! \brief Logger name for setup logger configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::StreamReaderFactory";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

private:
    //========================================
    //! Constructor registering all MVIS SDK configuration extensions.
    //!
    //! Configurations which are not delivered as plugins to the customer have to be registered manually.
    //!
    //! \note When adding new configurations with the sdk they have to be registered here.
    //----------------------------------------
    ConfigurationFactory();

public:
    //========================================
    //! \brief Get the singleton instance of ConfigurationFactory.
    //! \return Singleton instance of ConfigurationFactory.
    //----------------------------------------
    static ConfigurationFactory& getInstance();

public:
    //========================================
    //! \brief Get list of default parameter set ids.
    //! \param[in] configurationType  Configuration type.
    //! \return List with all possible default parameter set ids.
    //----------------------------------------
    const std::vector<std::string>& getDefaultParameterSets(const std::string& configurationType) const;

    //========================================
    //! \brief Create a configuration from type.
    //! \param[in] configurationType  Configuration type.
    //! \return Either \c Configuration pointer if registered, otherwise \c nullptr.
    //----------------------------------------
    ConfigurationPtr createConfiguration(const std::string& configurationType) const;

    //========================================
    //! \brief Create a configuration by configuration type.
    //! \param[in] configurationType    Unique human readable configuration type name string of the wanted configuration.
    //! \param[in] defaultParameterSet  Human readable identifier of default parameter set,
    //!                                 which will pass the default values by configuration constructor.
    //! \return Either a shared pointer to an instance of the \c Configuration or otherwise \c nullptr.
    //----------------------------------------
    ConfigurationPtr createConfiguration(const std::string& configurationType,
                                         const std::string& defaultParameterSet) const;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
