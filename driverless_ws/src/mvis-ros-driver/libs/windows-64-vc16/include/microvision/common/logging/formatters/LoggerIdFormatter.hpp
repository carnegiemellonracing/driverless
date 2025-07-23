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

#include <microvision/common/logging/formatters/FormatterWithPrecision.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief A log message formatter for converting the message's logger ID field.
//------------------------------------------------------------------------------
class LoggerIdFormatter : public FormatterWithPrecision
{
public: // constructors, destructors
    //========================================
    //! \brief Create an instance of a formatter for the message's logger ID field.
    //!
    //! \param[in] modifier  The \a FormatModifier used for applying common format options.
    //! \param[in] options   A list of strings with special format options for the logger ID field.
    //! \return An instance of a LoggerIdFormatter.
    //!
    //! Only the first entry in the options list is used. It is interpreted as an integer that limits the number of
    //! elements to be printed. Truncation is done from the beginning of the logger ID.
    //! E.g. the pattern “%logger{2}” will return the two rightmost elements of the logger ID only. So, if the ID is
    //! “microvision.common.logging.LogManager” it will return “logging.LogManager”.
    //! Default is to return the complete ID.
    //----------------------------------------
    static FormatterSPtr create(const FormatModifier& modifier, const std::list<std::string>& options);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~LoggerIdFormatter() = default;

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
    LoggerIdFormatter() = default;
}; // LoggerIdFormatter

//==============================================================================

using LoggerIdFormatterSPtr = std::shared_ptr<LoggerIdFormatter>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
