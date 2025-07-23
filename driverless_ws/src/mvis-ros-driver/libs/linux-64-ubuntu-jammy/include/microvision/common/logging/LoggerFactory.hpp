//==============================================================================
//! \file
//! \brief Factory produces logger frontends.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/Logger.hpp>
#include <microvision/common/logging/backends/LoggerBackendManager.hpp>

#include <map>
#include <memory>

//==============================================================================

#ifdef _WIN32
#    pragma warning(disable : 4251)
// class 'xxx' needs to have dll - interface to be used by clients of class 'yyy'
#endif

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief Central class for accessing logger frontends.
//------------------------------------------------------------------------------
class LoggerFactory final
{
public:
    //========================================
    //! \brief Constructor.
    //!
    //! \param[in] backendManager  Object that maintains the list of currently registered logging backends.
    //----------------------------------------
    LoggerFactory(LoggerBackendManagerSPtr backendManager) : m_backendManager(backendManager) {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~LoggerFactory();

public:
    //========================================
    //! \brief Create a logger with the given ID.
    //!
    //! \param[in] loggerId  ID of the logger to be created.
    //! \return If a logger with the same ID has been created before, a reference to this logger is returned,
    //!         otherwise a new logger is being created.
    //!
    //! \note After creation, the new logger is automatically configured using the current values set with \a configure
    //!       method.
    //----------------------------------------
    LoggerSPtr createLogger(const std::string& loggerId);

    //========================================
    //! \brief Update the configuration for logger frontends.
    //!
    //! \param[in] config  The new configuration.
    //!
    //! \note The new configuration is automatically populated to all currently active loggers and is used when new
    //!       loggers are created using the \a createLogger method.
    //----------------------------------------
    bool configure(const ConfigurationSPtr& config);

    //========================================
    //! \brief Set the default log level.
    //!
    //! \param[in] logLevel  The log level to be used for all loggers that do not have a specific configuration.
    //----------------------------------------
    void setDefaultLogLevel(const LogLevel logLevel);

    //========================================
    //! \brief Returns the default log level.
    //----------------------------------------
    LogLevel getDefaultLogLevel() const;

    //========================================
    //! \brief Get a list with all currently active loggers.
    //!
    //! \return List with currently active loggers.
    //----------------------------------------
    std::list<LoggerSPtr> getLoggers() const;

    //========================================
    //! \brief Get the currently active logging system configuration.
    //!
    //! \return Logging system configuration.
    //----------------------------------------
    ConfigurationSPtr getConfiguration() const;

private:
    using Mutex                  = std::mutex;
    using MutexGuard             = std::lock_guard<Mutex>;
    using LoggersByClassIdMap    = std::map<std::string, LoggerWPtr>;
    using BackendIdsByClassIdMap = std::map<std::string, std::string>;

private:
    void updateAllLoggerConfiguration();
    void updateLoggerConfiguration(const LoggerSPtr& logger);
    void cleanupLoggers();

private:
    LoggerBackendManagerSPtr m_backendManager{nullptr};
    mutable Mutex m_mutex{};
    LoggersByClassIdMap m_loggers{};
    BackendIdsByClassIdMap m_backendIds{};
    ConfigurationSPtr m_config{std::make_shared<Configuration>()};
}; // LoggerFactory

//==============================================================================

using LoggerFactorySPtr = std::shared_ptr<LoggerFactory>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
