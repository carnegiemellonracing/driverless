//==============================================================================
//! \file
//! \brief Configuration for backends and loggers.
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
#include <microvision/common/logging/Format.hpp>
#include <microvision/common/logging/backends/ConsoleLoggerBackend.hpp>
#include <microvision/common/logging/LoggingExport.hpp>

#include <string>
#include <list>
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
//! \brief Class holding the configuration of the logging system.
//!
//! The configuration is split into a part for specific loggers identified by their ID and a common part. Each part
//! contains a log level and a list of backends where the logger is connected to. Thus, an application can assign
//! dedicated log levels and backends for some loggers while the rest uses the common values. Using wildcard IDs
//! for specific loggers an application can assign values to a group of loggers in the same namespace.
//!
//! \note This class is not thread-safe. An application has to take special actions if instances are shared between
//!       different threads.
//------------------------------------------------------------------------------
class LOGGING_EXPORT Configuration
{
public:
    using ClassIdList = std::list<std::string>;

public:
    //========================================
    //! \brief Constructor.
    //!
    //! This constructor initializes a default configuration where the log level is set to \a LogLevel::Error and all
    //! loggers are connected to the \a ConsoleLoggerBackend only.
    //----------------------------------------
    Configuration() {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Configuration() {}

public: // getter
    //========================================
    //! \brief Get the log level for a given logger.
    //!
    //! \param[in] loggerId  The ID of the logger to retrieve the log level for.
    //! \return  The log level of the logger with the given ID.
    //!
    //! Precedence for getting the log level:
    //! <ul>
    //!   <li>First, look for a dedicated log level for the given logger ID.</li>
    //!   <li>Next, check if there is a dedicated log level for a wildcard ID matching the given logger ID. The
    //!       wildcard search finds the most specific match, i.e. \c microvision::common:logging::* has precedence over
    //!       \c microvision::common::* if both wildcard ID match.</li>
    //!   <li>Last, the configured default log level is used.</li>
    //! </ul>
    //----------------------------------------
    LogLevel getLogLevelForLogger(const std::string& loggerId);

    //========================================
    //! \brief Get the list of backend IDs for a given logger.
    //!
    //! \param[in] loggerId  The ID of the logger to retrieve the backend IDs for.
    //! \return  The list of backend IDs for the logger with the given ID.
    //!
    //! Precedence for getting the backend IDs:
    //! <ul>
    //!   <li>First, look for a dedicated backend ID list for the given logger ID.</li>
    //!   <li>Next, check if there is a dedicated backend ID list for a wildcard ID matching the given logger ID. The
    //!       wildcard search finds the most specific match, i.e. \c microvision::common:logging::* has precedence over
    //!       \c microvision::common::* if both wildcard ID match.</li>
    //!   <li>Last, the configured default backend ID list is used.</li>
    //! </ul>
    //----------------------------------------
    ClassIdList getBackendsForLogger(const std::string& loggerId);

public: // setter
    //========================================
    //! \brief Set the default log level.
    //!
    //! \param[in] logLevel  The log level to be used for all loggers that do not have a specific configuration.
    //----------------------------------------
    void setDefaultLogLevel(const char* const logLevel) { setDefaultLogLevel(parseLogLevel(logLevel)); }

    //========================================
    //! \brief Set the default log level.
    //!
    //! \param[in] logLevel  The log level to be used for all loggers that do not have a specific configuration.
    //----------------------------------------
    void setDefaultLogLevel(const LogLevel logLevel) { m_logLevel = logLevel; }

    //========================================
    //! \brief Returns the default log level.
    //----------------------------------------
    LogLevel getDefaultLogLevel() const { return m_logLevel; }

    //========================================
    //! \brief Set the default backend ID list.
    //!
    //! \param[in] backendIds  The IDs of the backends to be used for all loggers that do not have a specific
    //!                        configuration.
    //----------------------------------------
    void setDefaultBackends(const ClassIdList& backendIds);

    //========================================
    //! \brief Set the log level for a specific logger.
    //!
    //! \param[in] logLevel  The log level to set.
    //! \param[in] loggerId  The ID of the logger to set the given log level for. This ID might be a wildcard ID.
    //----------------------------------------
    void setLogLevelForLogger(const char* const logLevel, const std::string& loggerId);

    //========================================
    //! \brief Add a backend ID to the list of backend IDs for a specific logger.
    //!
    //! \param[in] backendId  The backend ID to add.
    //! \param[in] loggerId   The ID of the logger to add the given backend ID to. This ID might be a wildcard ID.
    //----------------------------------------
    void addBackendForLogger(const std::string& backendId, const std::string& loggerId);

private:
    using LogLevelMap   = std::map<std::string, LogLevel>;
    using BackendIdsMap = std::map<std::string, ClassIdList>;

private:
    static bool isWildcard(const std::string& loggerId);

    static bool matchesWildcard(const std::string& thisLoggerId, const std::string& otherLoggerId);

    template<typename TValue>
    static typename std::map<std::string, TValue>::const_iterator
    findLogger(const std::map<std::string, TValue>& theMap, const std::string& loggerId)
    {
        // Search by exact logger ID first.
        typename std::map<std::string, TValue>::const_iterator resultIter = theMap.find(loggerId);
        if (resultIter == theMap.end())
        {
            // Exact logger ID not found -> check wildcards.
            for (typename std::map<std::string, TValue>::const_iterator iter = theMap.begin(); iter != theMap.end();
                 ++iter)
            {
                if (isWildcard(iter->first) && (matchesWildcard(iter->first, loggerId)))
                {
                    // Potential match found -> is it better than the one before?
                    if ((resultIter == theMap.end()) || (iter->first.size() > resultIter->first.size()))
                    {
                        // No match found yet or more specific match found -> use it.
                        resultIter = iter;
                    }
                    // else: the current match is equal or worse than the previous one -> skip it.
                }
            }
        }

        return resultIter;
    }

private:
    static constexpr const char wildcardChar{'*'};

private:
    LogLevel m_logLevel{LogLevel::Error};
    ClassIdList m_backendIds{ConsoleLoggerBackend::getBackendId()};

    LogLevelMap m_loggerIds2LogLevel{};
    BackendIdsMap m_loggerIds2BackendIds{};
}; // Configuration

//==============================================================================

using ConfigurationSPtr = std::shared_ptr<Configuration>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
