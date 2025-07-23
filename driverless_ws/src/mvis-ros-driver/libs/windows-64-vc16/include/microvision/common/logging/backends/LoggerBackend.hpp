//==============================================================================
//! \file
//! \brief Base class for logger backends.
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
#include <microvision/common/logging/Format.hpp>
#include <microvision/common/logging/LoggingExport.hpp>

#include <tinyxml2.h>
#include <mutex>
#include <iostream>

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
//! \brief Base class for all logger backends.
//------------------------------------------------------------------------------
class LOGGING_EXPORT LoggerBackend
{
public:
    //========================================
    //! \brief Constructor.
    //!
    //! \param[in] backendId  The unique ID of this backend.
    //----------------------------------------
    LoggerBackend(const std::string& backendId) : m_backendId{backendId} {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~LoggerBackend() = default;

    //========================================
    //! \brief Log a message.
    //!
    //! \param[in] loggerId  ID of the logger which sent the message.
    //! \param[in] level     Log level as determined by the logger.
    //! \param[in] msg       The message to log.
    //----------------------------------------
    virtual void log(const std::string& loggerId, const LogLevel& level, const MessageSPtr msg) = 0;

    //========================================
    //! \brief Get the unique ID of this backend.
    //!
    //! \return  The ID of this backend.
    //----------------------------------------
    const std::string& getId() const { return m_backendId; }

    //========================================
    //! \brief Configure this backend from an XML node.
    //!
    //! \param[in] xmlNode         XML node containing the configuration.
    //! \param[in] suppressErrors  If set to \c true output of error messages shall be suppressed, otherwise they shall
    //!                            be printed to stderr.
    //! \return  \c true if the configuration was successful, \c false otherwise.
    //!
    //! \note Sample configuration inside the XML node:
    //! \code
    //! <Backend id="microvision::common::logging::FileLoggerBackend"
    //!          format="%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n">
    //!     <Path>microvision.log</Path>
    //! </Backend>
    //! \endcode
    //----------------------------------------
    virtual bool configure(const tinyxml2::XMLElement* const xmlNode, const bool suppressErrors);

    //========================================
    //! \brief Waits until all log messages are processed.
    //!
    //! \param[in] maxFlushTimeMilliseconds  Max. time in milliseconds to wait until all log messages are processed.
    //! \return \c true if all log messages have been processed, or \c false if a timeout occurred.
    //----------------------------------------
    virtual bool flush(const uint32_t = 100) const { return true; };

protected:
    std::string getFormattedText(const std::string& loggerId, const LogLevel level, const MessageSPtr msg)
    {
        const auto format = m_format; // make sure a reference stays alive for the formatting!
        return format->getFormattedText(loggerId, level, msg);
    }

protected:
    FormatSPtr m_format{Format::parse("")};
    std::string m_backendId;
}; // LoggerBackend

//==============================================================================

using LoggerBackendSPtr = std::shared_ptr<LoggerBackend>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
