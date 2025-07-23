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

#include <microvision/common/logging/LogLevel.hpp>
#include <microvision/common/logging/Message.hpp>
#include <microvision/common/logging/formatters/FormatModifier.hpp>

#include <string>
#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief Base class for all log message formatters.
//!
//! This (abstract) class acts as a base for all classes that do format conversions for log messages.
//------------------------------------------------------------------------------
class Formatter
{
public:
    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Formatter() = default;

public:
    //========================================
    //! \brief Get the format modifier of this formatter.
    //!
    //! \return The \a FormatModifier used to format the text created by this formatter.
    //----------------------------------------
    const FormatModifier& getModifier() const { return m_modifier; }

public:
    //========================================
    //! \brief Set the format modifier of this formatter.
    //!
    //! \param[in] modifier  The \a FormatModifier used to format the text.
    //----------------------------------------
    void setModifier(const FormatModifier& modifier) { m_modifier = modifier; }

public:
    //========================================
    //! \brief Format the message.
    //!
    //! \param[in] loggerId  ID of the logger that created the message.
    //! \param[in] level     The log level when the message was sent.
    //! \param[in] msg       The log message to be formatted.
    //! \return Text containing the log message formatted according to the configuration of the implmenting formatter.
    //----------------------------------------
    virtual std::string formatMessage(const std::string& loggerId, const LogLevel level, const MessageSPtr msg) = 0;

protected:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Formatter() = default;

protected:
    FormatModifier m_modifier;
}; // Formatter

//==============================================================================

//==============================================================================
//! \brief Type definition for a shared pointer to a \a Formatter.
//------------------------------------------------------------------------------
using FormatterSPtr = std::shared_ptr<Formatter>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
