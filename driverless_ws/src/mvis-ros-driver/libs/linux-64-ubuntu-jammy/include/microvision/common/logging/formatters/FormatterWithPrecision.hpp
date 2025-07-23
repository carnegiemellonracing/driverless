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
//! \date Nov 01, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/formatters/Formatter.hpp>

#include <cstdint>
#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief Base class for all log message formatters that use a precision option.
//------------------------------------------------------------------------------
class FormatterWithPrecision : public Formatter
{
public: // constructors, destructors
    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~FormatterWithPrecision() = default;

protected:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    FormatterWithPrecision() = default;

protected:
    //========================================
    //! \brief Get the precision value (integer) from the first option (if given).
    //!
    //! \param[in] options   A list of strings with special format options for the timestamp field.
    //----------------------------------------
    void storePrecisionFromOptions(const std::list<std::string>& options);

    //========================================
    //! \brief Format the given test with the precision value.
    //!
    //! \param[in] separator  String used to separate the elements in the texts string.
    //! \param[in] text       Text to be formatted.
    //! \return Text formatted to the number of elements given by the precision value.
    //----------------------------------------
    std::string applyPrecision(const std::string& separator, const std::string& text);

protected:
    uint32_t m_precision{0};
}; // FormatterWithPrecision

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
