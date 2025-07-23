//==============================================================================
//! \file
//! \brief Formatting of the messages.
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
#include <microvision/common/logging/LoggingExport.hpp>

#include <microvision/common/logging/formatters/FormatModifier.hpp>
#include <microvision/common/logging/formatters/Formatter.hpp>

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
//! \brief Class storing the format for log messages.
//!
//! This class is not implemented yet!
//------------------------------------------------------------------------------
class LOGGING_EXPORT Format
{
public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Format() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Format() = default;

    //========================================
    //! \brief Create a format object by parsing the given string.
    //!
    //! \param[in] formatString  The string containing the format description.
    //! \return The format as parsed from the string.
    //----------------------------------------
    static std::shared_ptr<Format> parse(const char* formatString);

    //========================================
    //! \brief Applies the format description set by the \a parse method to the given log message.
    //!
    //! \param[in] loggerId  ID of the logger that created the message.
    //! \param[in] level     The log level when the message was sent.
    //! \param[in] msg       The log message to be formatted.
    //! \return A text with the log message formatted to the format description.
    //----------------------------------------
    std::string getFormattedText(const std::string& loggerId, const LogLevel level, const MessageSPtr msg);

private:
    static std::list<std::string> parseFormatterOptions(const std::string& formatStr, size_t& pos);

private:
    using FormatterCreateFunction
        = FormatterSPtr (*)(const FormatModifier& modifier, const std::list<std::string>& options);
    using FormatterFactoryMap = std::map<std::string, FormatterCreateFunction>;

private:
    static const FormatterFactoryMap& getFormatterFactory();

protected:
    std::list<FormatterSPtr> m_formatters{};
}; // Format

//==============================================================================

using FormatSPtr = std::shared_ptr<Format>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
