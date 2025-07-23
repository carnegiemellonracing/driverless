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
//! \brief A log message formatter for converting the message's timestamp field.
//------------------------------------------------------------------------------
class DateFormatter : public Formatter
{
public: // constructors, destructors
    //========================================
    //! \brief Create an instance of a formatter for the message's timestamp field.
    //!
    //! \param[in] modifier  The \a FormatModifier used for applying common format options.
    //! \param[in] options   A list of strings with special format options for the timestamp field.
    //! \return An instance of a DateFormatter.
    //!
    //! Only the first entry in the options list is used. Valid patterns are:
    //! - DEFAULT: use default format (“%Y-%m-%d %H:%M:%S,%s”)
    //! - UNIX: print the number of seconds since start of epoch (1970-01-01 00:00:00).
    //! - UNIX_MILLIS: print the number of milliseconds since start of epoch (1970-01-01 00:00:00).
    //! - Any format string that can be interpreted by the strftime function. As a special extension the parameter %s
    //!   (lower case character) can be used here to print the milliseconds.
    //! If the option list is empty, the pattern "%x %X,%s" is used as default.
    //! A reference to the strftime format options can be found here:
    //! \a http://www.cplusplus.com/reference/ctime/strftime/
    //----------------------------------------
    static FormatterSPtr create(const FormatModifier& modifier, const std::list<std::string>& options);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~DateFormatter() = default;

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
    DateFormatter() = default;

private:
    std::string m_formatString{};
}; // DateFormatter

//==============================================================================

using DateFormatterSPtr = std::shared_ptr<DateFormatter>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
