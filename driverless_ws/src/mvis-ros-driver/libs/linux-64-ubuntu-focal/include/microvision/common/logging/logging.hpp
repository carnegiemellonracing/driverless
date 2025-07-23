//==============================================================================
//! \file
//! \brief Convenience header with all necessary includes and handy macro definitions.
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

#include <microvision/common/logging/backends/LoggerBackend.hpp>
#include <microvision/common/logging/backends/LoggerBackendManager.hpp>
#include <microvision/common/logging/backends/ConsoleLoggerBackend.hpp>
#include <microvision/common/logging/backends/FileLoggerBackend.hpp>

#include <microvision/common/logging/Configuration.hpp>
#include <microvision/common/logging/Format.hpp>

#include <microvision/common/logging/Logger.hpp>
#include <microvision/common/logging/LoggerFactory.hpp>
#include <microvision/common/logging/LogLevel.hpp>

#include <microvision/common/logging/Message.hpp>
#include <microvision/common/logging/MessageStreamHelper.hpp>
#include <microvision/common/logging/LogManager.hpp>

//==============================================================================
//! \brief Macro for easy creation of a log message with the current source code location.
//!
//! Typical use:
//! \code logger->critical(LOGMSG << "Logging an int: " << std::setw(4) << std::setfill('0') << 42); \endcode
//!
//! \note For logging a simple string without using the streaming output operator ('<<') the \a LOGMSG_TEXT macro
//!       should be considered which runs faster.
//------------------------------------------------------------------------------
#define LOGMSG std::make_shared<microvision::common::logging::Message>(__LINE__, __FUNCTION__, __FILE__)

//==============================================================================
//! \brief Macro for easy creation of a log message with the current source code location and a simple string.
//!
//! Typical use:
//! \code logger->critical(LOGMSG_TEXT("This is a log message!")); \endcode
//!
//! \note This is the preferred way for logging a simple string only. Nevertheless, you can still concatenate the log
//!       message with the streaming output operator ('<<') as shown in \a LOGMSG.
//------------------------------------------------------------------------------
#define LOGMSG_TEXT(text)                                                                                              \
    std::make_shared<microvision::common::logging::Message>(__LINE__, __FUNCTION__, __FILE__, text)

//==============================================================================
//! \brief Macros for easy creation of a log messages with a specific log level.
//!
//! Typical use:
//! \code LOGERROR(logger, "This is a log message!"); \endcode
//------------------------------------------------------------------------------
#define LOGCRITICAL(logger, stream)                                                                                    \
    {                                                                                                                  \
        if (logger && logger->isCritical())                                                                            \
        {                                                                                                              \
            logger->critical(LOGMSG << stream);                                                                        \
        }                                                                                                              \
    }
#define LOGERROR(logger, stream)                                                                                       \
    {                                                                                                                  \
        if (logger && logger->isError())                                                                               \
        {                                                                                                              \
            logger->error(LOGMSG << stream);                                                                           \
        }                                                                                                              \
    }
#define LOGWARNING(logger, stream)                                                                                     \
    {                                                                                                                  \
        if (logger && logger->isWarning())                                                                             \
        {                                                                                                              \
            logger->warning(LOGMSG << stream);                                                                         \
        }                                                                                                              \
    }
#define LOGINFO(logger, stream)                                                                                        \
    {                                                                                                                  \
        if (logger && logger->isInfo())                                                                                \
        {                                                                                                              \
            logger->info(LOGMSG << stream);                                                                            \
        }                                                                                                              \
    }
#define LOGTRACE(logger, stream)                                                                                       \
    {                                                                                                                  \
        if (logger && logger->isTrace())                                                                               \
        {                                                                                                              \
            logger->trace(LOGMSG << stream);                                                                           \
        }                                                                                                              \
    }
#define LOGDEBUG(logger, stream)                                                                                       \
    {                                                                                                                  \
        if (logger && logger->isDebug())                                                                               \
        {                                                                                                              \
            logger->debug(LOGMSG << stream);                                                                           \
        }                                                                                                              \
    }
