//==============================================================================
//! \file
//! \brief Logger backend that writes all messages to std::cout / std::cerr.
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

#include <microvision/common/logging/backends/AsyncLoggerBackend.hpp>
#include <microvision/common/logging/LogLevel.hpp>
#include <microvision/common/logging/Message.hpp>
#include <microvision/common/logging/LoggingExport.hpp>

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
//! \brief A logger backend for writing messages to the console.
//------------------------------------------------------------------------------
class LOGGING_EXPORT ConsoleLoggerBackend : public AsyncLoggerBackend
{
public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    ConsoleLoggerBackend() : AsyncLoggerBackend{getBackendId()} {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~ConsoleLoggerBackend();

public: // getter
    //========================================
    //! \brief The ID of this logger backend (static variant).
    //----------------------------------------
    static std::string getBackendId() { return backendId; }

public:
    //========================================
    //! \brief Log a message.
    //!
    //! \param[in] loggerId  ID of the logger which sent the message.
    //! \param[in] level     Log level as determined by the logger.
    //! \param[in] msg       The message to log.
    //!
    //! \note Messages with level \a LogLevel::Critical or \a LogLevel::Error are printed to stderr, others to stdout.
    //----------------------------------------
    void logAsync(const std::string& loggerId, const LogLevel& level, const MessageSPtr msg) override;

private:
    //========================================
    //! \brief The ID of this logger backend.
    //----------------------------------------
    constexpr static const char* const backendId{"microvision::common::logging::ConsoleLoggerBackend"};

    //========================================
    //! \brief Mutex to prevent logging multiple messages to the console simultaneously.
    //----------------------------------------
    mutable Mutex m_consoleMutex{};
}; // ConsoleLoggerBackend

//==============================================================================

using DefaultLoggerBackendSPtr = std::shared_ptr<ConsoleLoggerBackend>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
