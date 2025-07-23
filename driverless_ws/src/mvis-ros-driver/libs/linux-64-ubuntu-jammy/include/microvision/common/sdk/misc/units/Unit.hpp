//==============================================================================
//! \file
//!
//! \brief Contains helper functions for converting to SI units.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 24, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace unit {
//==============================================================================

//==============================================================================

//! conversion struct template
template<uint64_t UnitTypeFrom,
         uint64_t UnitTypeTo,
         typename T,
         typename = typename std::enable_if<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, T>::type>
struct Convert
{
    //========================================
    //!\brief do the conversion from unit to unit for an input value
    //!\param[in]      val       value in unit from which should be converted
    //!\return the value converted into the second unit
    //!\note syntax of the call is Convert<UnitTypeFrom, UnitTypeTo, T>()(value)
    //----------------------------------------
    constexpr T operator()(const T& val) const = delete;
};

//! template for unit identifier and symbol
template<uint64_t UnitType>
struct Unit
{
    //========================================
    //!\brief get the name of the unit type
    //!\return name text of the unit
    // default is deleted here to only allow existing specializations
    //========================================
    constexpr static const char* name() = delete;

    //========================================
    //!\brief get the symbol of the unit type
    //!\return symbol text of the unit
    // default is deleted here to only allow existing specializations
    //========================================
    constexpr static const char* symbol() = delete;
};

//==============================================================================

//! general constants
constexpr uint32_t mega = 1000000;
constexpr uint32_t kilo = 1000;

constexpr uint32_t deci  = 10;
constexpr uint32_t centi = 100;
constexpr uint32_t milli = 1000;
constexpr uint32_t micro = 1000000;
constexpr uint32_t nano  = 1000000000;

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
