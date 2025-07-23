//==============================================================================
//! \file
//! \brief All available log levels.
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

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <stdint.h>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//! Enumeration for supported log levels

//==============================================================================
//! \brief Log levels as supported by the logging system.
//------------------------------------------------------------------------------
enum class LogLevel : uint16_t
{
    //! Logging is turned off.
    Off = 0,
    //! Critical error, task cannot continue, application/service will stop.
    Critical = 10,
    //! Task cannot continue but application/service can still function.
    Error = 20,
    //! Undesirable or unexpected runtime situation - progressive failure possible.
    Warning = 30,
    //! General flow of application/service.
    Info = 40,
    //! Most detailed information, e.g. entering/leaving function.
    Trace = 50,
    //! Detailed information, e.g object or function parameters.
    Debug = 60
}; // LogLevel

//==============================================================================
//! \brief Convert a string into a log level.
//------------------------------------------------------------------------------
inline LogLevel parseLogLevel(const char* const extLogLevel)
{
    LogLevel logLevel{LogLevel::Off};
    std::string extLogLevelStr{extLogLevel};

    // Convert to lower case (for case-insensitive comparison).
    std::transform(extLogLevelStr.begin(), extLogLevelStr.end(), extLogLevelStr.begin(), [](const unsigned char i) {
        return std::tolower(i);
    });

    if (extLogLevelStr == "critical")
    {
        logLevel = LogLevel::Critical;
    }
    else if (extLogLevelStr == "error")
    {
        logLevel = LogLevel::Error;
    }
    else if (extLogLevelStr == "warning")
    {
        logLevel = LogLevel::Warning;
    }
    else if (extLogLevelStr == "info")
    {
        logLevel = LogLevel::Info;
    }
    else if (extLogLevelStr == "trace")
    {
        logLevel = LogLevel::Trace;
    }
    else if (extLogLevelStr == "debug")
    {
        logLevel = LogLevel::Debug;
    }

    return logLevel;
}

//==============================================================================
//! \brief Convert a log level into a string.
//------------------------------------------------------------------------------
inline std::string logLevelToString(LogLevel logLevel)
{
    switch (logLevel)
    {
    case LogLevel::Off:
        return "Off";
    case LogLevel::Critical:
        return "Critical";
    case LogLevel::Error:
        return "Error";
    case LogLevel::Warning:
        return "Warning";
    case LogLevel::Info:
        return "Info";
    case LogLevel::Trace:
        return "Trace";
    case LogLevel::Debug:
        return "Debug";

    default:
        throw std::invalid_argument("Unknown log level!");
    }
}

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
