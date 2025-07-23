//==============================================================================
//! \file
//! \brief Frontend used for logging.
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

#include <microvision/common/logging/LogLevel.hpp>
#include <microvision/common/logging/Message.hpp>
#include <microvision/common/logging/backends/LoggerBackend.hpp>
#include <microvision/common/logging/LoggingExport.hpp>

#include <string>
#include <thread>
#include <mutex>
#include <list>

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
//! \brief Class for sending messages into the logging system.
//!
//! An application uses instances of this class for sending log messages which are filtered by the individual
//! \a LogLevel. Additionally, each logger has a list of connected backends used for actually storing or visualizing
//! the messages.
//!
//! Instances of this class should be created via the \a LoggerFactory.
//------------------------------------------------------------------------------
class LOGGING_EXPORT Logger
{
public:
    //========================================
    //! \brief This constructor creates an empty logger which is not connected to any backend and with a log level set
    //! to \a LogLevel::Off.
    //!
    //! \param[in] loggerId  Unique ID of this logger.
    //!
    //! \note Loggers should not be created directly using this constructor but via the \a LoggerFactory instead.
    //----------------------------------------
    Logger(const std::string& loggerId);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Logger();

public:
    //========================================
    //! \brief Get the unique ID of this logger.
    //!
    //! \return The unique ID of this logger.
    //----------------------------------------
    const std::string& getId() const { return m_loggerId; }

public:
    //========================================
    //! \brief Add a backend to this logger.
    //!
    //! \param[in] loggerBackend  Backend to be connected to this logger.
    //----------------------------------------
    void addLoggerBackend(LoggerBackendSPtr loggerBackend);

    //========================================
    //! \brief Remove a backend from this logger.
    //!
    //! \param[in] loggerBackendId  ID of the backend to be removed from this logger.
    //----------------------------------------
    void removeLoggerBackend(const std::string& loggerBackendId);

    //========================================
    //! \brief Remove all backends from this logger.
    //!
    //! This method clears the internal list of connected backends.
    //----------------------------------------
    void removeAllBackends();

    //========================================
    //! \brief Get a list of backend IDs where this logger is connected to.
    //!
    //! \return A list with IDs of connected backends.
    //----------------------------------------
    std::list<std::string> getBackendIds() const;

    //========================================
    //! \brief Get the log level for this logger.
    //!
    //! \return The log level that was set for this logger.
    //----------------------------------------
    LogLevel getLogLevel() const;

    //========================================
    //! \brief Set the log level for this logger.
    //!
    //! \param[in] level  The new log level to be set.
    //----------------------------------------
    void setLogLevel(const LogLevel level);

public:
    //========================================
    //! \brief Send a log message into the logging system.
    //!
    //! \param[in] level  The level to be used for this message.
    //! \param[in] msg    The message to be sent.
    //!
    //! The logger sends this message to all connected backends.
    //----------------------------------------
    void log(const LogLevel level, const MessageSPtr msg);

    //========================================
    //! \brief Send a log message with level \a LogLevel::Critical into the logging system.
    //!
    //! \param[in] msg  The message to be sent.
    //----------------------------------------
    void critical(const MessageSPtr msg) { log(LogLevel::Critical, msg); }

    //========================================
    //! \brief Send a log message with level \a LogLevel::Error into the logging system.
    //!
    //! \param[in] msg  The message to be sent.
    //----------------------------------------
    void error(const MessageSPtr msg) { log(LogLevel::Error, msg); }

    //========================================
    //! \brief Send a log message with level \a LogLevel::Warning into the logging system.
    //!
    //! \param[in] msg  The message to be sent.
    //----------------------------------------
    void warning(const MessageSPtr msg) { log(LogLevel::Warning, msg); }

    //========================================
    //! \brief Send a log message with level \a LogLevel::Info into the logging system.
    //!
    //! \param[in] msg  The message to be sent.
    //----------------------------------------
    void info(const MessageSPtr msg) { log(LogLevel::Info, msg); }

    //========================================
    //! \brief Send a log message with level \a LogLevel::Trace into the logging system.
    //!
    //! \param[in] msg  The message to be sent.
    //----------------------------------------
    void trace(const MessageSPtr msg) { log(LogLevel::Trace, msg); }

    //========================================
    //! \brief Send a log message with level \a LogLevel::Debug into the logging system.
    //!
    //! \param[in] msg  The message to be sent.
    //----------------------------------------
    void debug(const MessageSPtr msg) { log(LogLevel::Debug, msg); }

    //========================================
    //! \brief Check whether the given log level is active.
    //!
    //! \param[in] level  Log level to check.
    //! \return \c true if the given level is equally or more important than the log level currently set for this
    //!         logger (see \a getLogLevel), or \c false otherwise.
    //----------------------------------------
    bool isLogLevelActive(const LogLevel& level) const;

    //========================================
    //! \brief Check whether the \a LogLevel::Critical is active.
    //!
    //! \return \c true if currently set log level for this logger is \a LogLevel::Critical or more detailed.
    //----------------------------------------
    bool isCritical() const { return isLogLevelActive(LogLevel::Critical); }

    //========================================
    //! \brief Check whether the \a LogLevel::Error is active.
    //!
    //! \return \c true if currently set log level for this logger is \a LogLevel::Error or more detailed.
    //----------------------------------------
    bool isError() const { return isLogLevelActive(LogLevel::Error); }

    //========================================
    //! \brief Check whether the \a LogLevel::Warning is active.
    //!
    //! \return \c true if currently set log level for this logger is \a LogLevel::Warning or more detailed.
    //----------------------------------------
    bool isWarning() const { return isLogLevelActive(LogLevel::Warning); }

    //========================================
    //! \brief Check whether the \a LogLevel::Info is active.
    //!
    //! \return \c true if currently set log level for this logger is \a LogLevel::Info or more detailed.
    //----------------------------------------
    bool isInfo() const { return isLogLevelActive(LogLevel::Info); }

    //========================================
    //! \brief Check whether the \a LogLevel::Debug is active.
    //!
    //! \return \c true if currently set log level for this logger is \a LogLevel::Debug or more detailed.
    //----------------------------------------
    bool isDebug() const { return isLogLevelActive(LogLevel::Debug); }

    //========================================
    //! \brief Check whether the \a LogLevel::Trace is active.
    //!
    //! \return \c true if currently set log level for this logger is \a LogLevel::Trace or more detailed.
    //----------------------------------------
    bool isTrace() const { return isLogLevelActive(LogLevel::Trace); }

    //========================================
    //! \brief Check whether the logging is switched on.
    //!
    //! \return \c true if logging is switched on (i.e. the currently set log level for this logger is not
    //!         \a LogLevel::Off), or \c false otherwise.
    //----------------------------------------
    bool isEnabled() const { return !isLogLevelActive(LogLevel::Off); }

private:
    using Mutex      = std::recursive_mutex;
    using MutexGuard = std::lock_guard<Mutex>;

private:
    std::list<LoggerBackendSPtr>::const_iterator findBackend(const std::string& backendId) const;

private:
    mutable Mutex m_mutex{};
    std::string m_loggerId;
    LogLevel m_logLevel{LogLevel::Off};
    std::list<LoggerBackendSPtr> m_loggerBackends{};
}; // Logger

//==============================================================================

using LoggerSPtr = std::shared_ptr<Logger>;
using LoggerWPtr = std::weak_ptr<Logger>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
