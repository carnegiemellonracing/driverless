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
#include <microvision/common/logging/formatters/FormatModifier.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief A log message formatter for converting the message's file path field.
//------------------------------------------------------------------------------
class FilePathFormatter : public FormatterWithPrecision
{
public:
#ifdef _WIN32
    constexpr static char pathSeparator = '\\';
#else
    constexpr static char pathSeparator = '/';
#endif

public: // constructors, destructors
    //========================================
    //! \brief Create an instance of a formatter for the message's file path field.
    //!
    //! \param[in] modifier  The \a FormatModifier used for applying common format options.
    //! \param[in] options   A list of strings with special format options for the file path field.
    //! \return An instance of a FilePathFormatter.
    //!
    //! Only the first entry in the options list is used. It is interpreted as an integer that limits the number of
    //! elements to be printed. Truncation is done from the beginning of the file path.
    //! E.g. the pattern “%file{2}” will return the two rightmost elements of the path only. So, if the path is
    //! “foo1/foo2/bar.cpp” it will return “foo2/bar.cpp”.
    //! Default is to return the complete path.
    //----------------------------------------
    static FormatterSPtr create(const FormatModifier& modifier, const std::list<std::string>& options);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~FilePathFormatter() = default;

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
    FilePathFormatter() = default;

private:
}; // FilePathFormatter

//==============================================================================

using FilePathFormatterSPtr = std::shared_ptr<FilePathFormatter>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
