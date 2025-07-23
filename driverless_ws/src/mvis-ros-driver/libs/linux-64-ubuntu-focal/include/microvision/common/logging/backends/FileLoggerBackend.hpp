//==============================================================================
//! \file
//! \brief File logger backend - writes all messages to file.
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

#include <fstream>
#include <string>
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
//! \brief A logger backend for writing messages to a file.
//------------------------------------------------------------------------------
class LOGGING_EXPORT FileLoggerBackend : public AsyncLoggerBackend
{
public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    FileLoggerBackend();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~FileLoggerBackend();

public: // getter
    //========================================
    //! \brief The ID of this logger backend (static variant).
    //----------------------------------------
    static std::string getBackendId() { return backendId; }

    //========================================
    //! \brief Get the path to the log file.
    //----------------------------------------
    const std::string& getFilePath() const;

public: // setter
    //========================================
    //! \brief Set the path of the log file.
    //!
    //! \param[in] filePath  Path of the file where the log messages are written to.
    //----------------------------------------
    void setFilePath(const std::string& filePath);

public:
    //========================================
    //! \brief Log a message.
    //!
    //! \param[in] loggerId  ID of the logger which sent the message.
    //! \param[in] level     Log level as determined by the logger.
    //! \param[in] msg       The message to log.
    //----------------------------------------
    void logAsync(const std::string& loggerId, const LogLevel& level, const MessageSPtr msg) override;

    //========================================
    //! \brief Configure this backend from an XML node.
    //!
    //! \param[in] xmlNode         XML node containing the configuration.
    //! \param[in] suppressErrors  If set to \c true output of error messages shall be suppressed, otherwise they shall
    //!                            be printed to stderr.
    //! \return  \c true if the configuration was successful, \c false otherwise.
    //----------------------------------------
    bool configure(const tinyxml2::XMLElement* const xmlNode, const bool suppressErrors) override;

private:
    static std::string expandEnvironmentVariables(const std::string& input);
    constexpr static const char* const backendId{"microvision::common::logging::FileLoggerBackend"};

    std::fstream m_logFile;
    bool m_logFileOpenFailed{false};
    std::string m_filePath{"default.log"};
}; // FileLoggerBackend

//==============================================================================

using FileLoggerBackendSPtr = std::shared_ptr<FileLoggerBackend>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
