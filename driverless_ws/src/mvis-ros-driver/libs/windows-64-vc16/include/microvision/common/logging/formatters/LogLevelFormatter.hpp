//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Oct 29, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/formatters/Formatter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief A log message formatter for converting the message's log level field.
//------------------------------------------------------------------------------
class LogLevelFormatter : public Formatter
{
public: // constructors, destructors
    //========================================
    //! \brief Create an instance of a formatter for the message's log level field.
    //!
    //! \param[in] modifier  The \a FormatModifier used for applying common format options.
    //! \param[in] options   A list of strings with special format options for the log level field.
    //! \return An instance of a LogLevelFormatter.
    //!
    //! This formatter uses the following options from the options list:
    //! - "length" is used to limit the length of the output (default is no limit).
    //! - "lowerCase" is used to convert the output to lower case, if set to \c true, or upper case otherwise
    //!   (default is upper case).
    //! E.g. the pattern “%logLevel{length=1}{lowerCase=true}” will return the the first character of the log level in
    //! lower case only. So, if the log level is “DEBUG” it will return “d”.
    //----------------------------------------
    static FormatterSPtr create(const FormatModifier& modifier, const std::list<std::string>& options);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~LogLevelFormatter() = default;

public:
    //========================================
    //! \brief Format the message.
    //!
    //! \param[in] loggerId  ID of the logger that created the message.
    //! \param[in] level     The log level when the message was sent.
    //! \param[in] msg       The log message to be formatted.
    //! \return Text containing the log message formatted according to the configuration of this formatter.
    //----------------------------------------
    std::string formatMessage(const std::string& loggerId, const LogLevel level, const MessageSPtr msg) override;

private:
    LogLevelFormatter() = default;

private:
    uint32_t m_length{std::numeric_limits<uint32_t>::max()};
    bool m_lowerCase{false};
}; // LogLevelFormatter

//==============================================================================

using LogLevelFormatterSPtr = std::shared_ptr<LogLevelFormatter>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
