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

#include <limits>
#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief Class for handling the format modifiers of a formatter element.
//------------------------------------------------------------------------------
class FormatModifier
{
public: // constructors, destructors
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    FormatModifier();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~FormatModifier() = default;

    //========================================
    //! \brief Creates a new instance of this class by parsing the given string at the given position.
    //!
    //! \param[in] formatStr  String with the format modifier representation.
    //! \param[in] pos        Position in the format string where parsing starts.
    //! \return The format modifier instance.
    //!
    //! Valid modifier patterns are (e.g. when formatting the message text):
    //!                  align       min. width       max. width
    //!   %20msg         right           20              none
    //!   %-20msg        left            20              none
    //!   %.30msg         n/a           none              30         truncate from left
    //!   %.-30msg        n/a           none              30         truncate from right
    //!   %20.30msg      right           20               30         truncate from left
    //!   %20.-30msg     right           20               30         truncate from right
    //!   %-20.30msg     left            20               30         truncate from left
    //!   %-20.-30msg    left            20               30         truncate from right
    //----------------------------------------
    static FormatModifier parse(const std::string& formatStr, std::size_t& pos);

public:
    //========================================
    //! \brief Apply the format modifications on the given text.
    //!
    //! \param[in] text  Text to format.
    //! \return Modified text according to the specification.
    //----------------------------------------
    std::string process(const std::string& text);

private:
    bool m_skip{true};
    bool m_padLeft{true};
    bool m_truncateLeft{true};
    uint32_t m_minWidth{0};
    uint32_t m_maxWidth;
}; // FormatModifier

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
