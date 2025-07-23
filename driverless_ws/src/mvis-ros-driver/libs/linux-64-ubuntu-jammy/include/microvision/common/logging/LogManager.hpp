//==============================================================================
//! \file
//! \brief Central access point for logging package.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 20, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/Logger.hpp>
#include <microvision/common/logging/Configuration.hpp>
#include <microvision/common/logging/LoggingExport.hpp>
#include <microvision/common/logging/LoggerFactory.hpp>
#include <microvision/common/logging/backends/LoggerBackendManager.hpp>

#include <string>
#include <memory>
#include <tinyxml2.h>

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
// Grant the unittest fixture access to the LogManager.
//==============================================================================
namespace unittest {
class Fixture;
} // namespace unittest
//==============================================================================

//==============================================================================
//! \brief Central access point for logging package.
//!
//! The \a LogManager class is used as central entry point for all topics regarding logging. It is implemented as
//! singleton. Applications can use this class by calling the (static) \a getInstance method.
//! The logging package is separated between the frontend and the backend.
//!
//! <h2>Frontend</h2>
//! An application uses \a Logger instances from the frontend to send messages of a given severity defined by the
//! \a LogLevel to the logging system. To create a logger, the application chooses a unique identity which is
//! formatted like a C++ class name with namespace: e.g. \c microvision::common::logging::Logger and calls either the
//! \a createLogger or \a createSimpleLogger method. The number of loggers an application can create is not
//! limited by the logging system. So, it is possible use a separate logger for each class or even for each class
//! instance (e.g. if the class name is suffixed by an index or ID). The more loggers an application uses, the finer
//! the logging can be controlled (see logging configuration).
//!
//! A special logger is the global logger which can be retrieved using the (static) \a globalLogger method. This
//! instance can be used to quickly log messages without worrying about creating a logger. The downside is, that only
//! one log level can be set for all messages sent through this logger.
//!
//! <h2>Backend</h2>
//! The backend part of the logging system receives the log messages, filters them according to the currently set log
//! level, and processes them. There are predefined logging backends for printing the log messages to the console
//! (\a ConsoleLoggerBackend) or to file (\a FileLoggerBackend). Each backend is identified by a unique class ID. An
//! application can create additional backends (e.g. for writing log messages into GUI window or a database) by
//! implementing the \a LoggerBackend interface, choosing an appropriate class ID, and registering it at the logging
//! system through the \a registerBackend method.
//!
//! <h2>Configuration</h2>
//! The properties of the logging system can be set by calling the method \a loadConfig with the path to an XML file
//! containing the configuration. Alternatively, the configuration can be set directly by providing an XML formatted
//! string through the \a parseConfig method. A typical configuration looks a follows:
//! \code
//! <Configuration>
//!     <Backends>
//!         <Backend id="microvision::common::logging::ConsoleLoggerBackend"
//!                  format="%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n" />
//!         <Backend id="microvision::common::logging::FileLoggerBackend"
//!                  format="%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n">
//!             <Path>microvision.log</Path>
//!         </Backend>
//!     </Backends>
//!     <Loggers>
//!         <Root level="debug">
//!             <BackendRef id="microvision::common::logging::ConsoleLoggerBackend" />
//!             <BackendRef id="microvision::common::logging::FileLoggerBackend" />
//!         </Root>
//!         <Logger id="microvision::common::logging::TestLogger" level="critical">
//!             <BackendRef id="microvision::common::logging::ConsoleLoggerBackend" />
//!         </Logger>
//!     </Loggers>
//! </Configuration>
//! \endcode
//! At startup the system tries to load the configuration from a file named "logconfig.xml" in the current working
//! directory. If this fails, the system uses a default configuration.
//!
//! <h3>Backend Configuration</h3>
//! The first section "Backends" is used to configure backend parameter. Each backend has a separate subsection
//! "Backend" and is identified by its attribute "id". A common parameter for all backends is the message format
//! (attribute "format", see \a Format). NOT IMPLEMENTED YET! Inside the "Backend" subsection there are additional
//! parameters, e.g. the file path for the \a FileLoggerBackend. The configuration for custom backends should follow
//! the same concept. The log manager takes care that each additional backend that is registered through the
//! \a registerBackend method is provided with the contents of the "Backend" subsection. Thus, individual parameters
//! can be defined for each custom backends inside its "Backend" subsection.
//!
//! <h3>Logger Configuration</h3>
//! The next section "Loggers" is used to configure the frontend. Each logger has its own subsection identified by the
//! "id" attribute. The log level can be set through the "level" attribute. Inside the logger subsection there is a
//! list of IDs referencing the backends this logger is connected with. So, it is possible to use the file and console
//! backend for one logger while another logger is logging to console only. Custom backends are referenced the same
//! way using their unique IDs.
//!
//! The "Root" subsection does not have an ID and has a special role. The entries in this section are used for loggers
//! that do not have a special configuration inside a "Logger" subsection.
//!
//! If the class name part of the logger ID is equal to the wildcard character "*", the configuration is valid for all
//! loggers of the given namespace.
//------------------------------------------------------------------------------
class LOGGING_EXPORT LogManager final
{
    friend class unittest::Fixture;

public:
    using ClassIdList = std::list<std::string>;

public:
    //========================================
    //! \brief Constructor (singleton).
    //!
    //! \return The one-and-only instance of the LogManager.
    //----------------------------------------
    static LogManager& getInstance();

    //========================================
    //! \brief Destructor
    //----------------------------------------
    ~LogManager();

public:
    //========================================
    //! \brief Create a new logger.
    //!
    //! This method creates a new logger with the given ID. The log level and and the connections to the backends are
    //! set as specified in the configuration (see \a loadConfig or \a parseConfig). Calling this method is the same
    //! as calling the logger factory directly with \a LoggerFactory::getInstance().createLogger(loggerId).
    //!
    //! \param[in] loggerId  ID of the logger to create.
    //! \return Logger created by the logging system.
    //----------------------------------------
    LoggerSPtr createLogger(const std::string& loggerId);

    //========================================
    //! \brief Get a list with all currently active loggers.
    //!
    //! \return List with currently active loggers.
    //----------------------------------------
    std::list<LoggerSPtr> getLoggers() const;

    //========================================
    //! \brief Register a new backend.
    //!
    //! This method is used to register a custom backend at the logging system. Without this registration the backend
    //! is not available for loggers. The built-in backends \a ConsoleLoggerBackend and \a FileLoggerBackend are
    //! registered automatically. Calling this method is the same as calling the logger backend manager directly with
    //! \a LoggerBackendManager::getInstance().registerBackend(backend);
    //!
    //! \param[in] backend  The new backend to be registered.
    //! \return \c true if the registration was successful, \c false otherwise.
    //!
    //! \note
    //! It is important to register custom backends before the logging system is configured (see \a loadConfig or
    //! \a parseConfig). Otherwise the backend will not be configured correctly.
    //----------------------------------------
    bool registerBackend(LoggerBackendSPtr backend);

    //========================================
    //! \brief Get a backend by its ID.
    //!
    //! \param[in] backendId  ID of the backend to look for.
    //! \return  The backend if the given ID was found, or an empty shared pointer (\c nullptr) otherwise.
    //----------------------------------------
    LoggerBackendSPtr getBackendById(const std::string& backendId);

    //========================================
    //! \brief Load logging system configuration.
    //!
    //! This method configures the logging system with the contents of an XML file with the given path.
    //!
    //! \param[in] path            Path to the configuration file.
    //! \param[in] suppressErrors  If set to \c true output of error messages are being suppressed, otherwise they are
    //!                            printed to stderr.
    //! \return \c true if the configuration was successful, \c false otherwise.
    //----------------------------------------
    bool loadConfig(const std::string& path, const bool suppressErrors = false);

    //========================================
    //! \brief Configure logging system.
    //!
    //! This method configures the logging system with the given XML formatted text.
    //!
    //! \param[in] xml             XML formatted string containing the logging system configuration.
    //! \param[in] suppressErrors  If set to \c true output of error messages are being suppressed, otherwise they are
    //!                            printed to stderr.
    //! \return \c true if the configuration was successful, \c false otherwise.
    //----------------------------------------
    bool parseConfig(const std::string& xml, const bool suppressErrors = false);

    //========================================
    //! \brief Configure logging system.
    //!
    //! This method configures the logging system with the given XML formatted text.
    //!
    //! \param[in] xml             XML formatted string containing the logging system configuration.
    //! \param[in] suppressErrors  If set to \c true output of error messages are being suppressed, otherwise they are
    //!                            printed to stderr.
    //! \return \c true if the configuration was successful, \c false otherwise.
    //----------------------------------------
    bool configure(const ConfigurationSPtr& config, const bool suppressErrors = false);

    //========================================
    //! \brief Set the default log level.
    //!
    //! \param[in] logLevel  The log level to be used for all loggers that do not have a specific configuration.
    //----------------------------------------
    void setDefaultLogLevel(const LogLevel logLevel);

    //========================================
    //! \brief Set the default log level.
    //!
    //! \param[in] logLevel  The log level to be used for all loggers that do not have a specific configuration.
    //----------------------------------------
    void setDefaultLogLevel(const char* const logLevel) { setDefaultLogLevel(parseLogLevel(logLevel)); }

    //========================================
    //! \brief Get the default log level.
    //!
    //! \return The log level to be used for all loggers that do not have a specific configuration.
    //----------------------------------------
    LogLevel getDefaultLogLevel() const;

    //========================================
    //! \brief Set the default backend ID list.
    //!
    //! \param[in] backendIds  The IDs of the backends to be used for all loggers that do not have a specific
    //!                        configuration.
    //----------------------------------------
    void setDefaultBackends(const ClassIdList& backendIds);

    //========================================
    //! \brief Get the currently active logging system configuration.
    //!
    //! \return Logging system configuration.
    //----------------------------------------
    ConfigurationSPtr getConfiguration() const;

    //========================================
    //! \brief Waits until all backends have processed all log messages.
    //!
    //! \param[in] maxFlushTimeMilliseconds  Max. time in milliseconds to wait until.
    //----------------------------------------
    void flushBackends(const uint32_t maxFlushTimeMilliseconds = 100) const;

public:
    //========================================
    //! \brief Get the global logger.
    //!
    //! This method returns the global logger singleton instance which can be used for logging general messages.
    //!
    //! \return The global logger singleton instance.
    //----------------------------------------
    static LoggerSPtr globalLogger();

private:
    using Mutex      = std::mutex;
    using MutexGuard = std::lock_guard<Mutex>;

private:
    LogManager();

private:
    void reset(); // For unit tests only.
    void registerDefaultBackends();

private:
    void parseBackendsConfig(const tinyxml2::XMLElement* const backendsNode, const bool suppressErrors);
    void parseLoggersConfig(ConfigurationSPtr& config,
                            const tinyxml2::XMLElement* const loggersNode,
                            const bool suppressErrors);
    std::list<std::string> parseBackendReferences(const std::string& loggerId,
                                                  const tinyxml2::XMLElement* loggerNode,
                                                  const bool suppressErrors);

private:
    LoggerFactorySPtr m_loggerFactory{nullptr};
    LoggerBackendManagerSPtr m_backendManager{nullptr};
    mutable Mutex m_globalLoggerMutex{};
    LoggerSPtr m_globalLogger{nullptr};
}; // LogManager

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
