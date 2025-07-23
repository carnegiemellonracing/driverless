//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 29, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <limits>
#include <cassert>
#include <cmath>

//==============================================================================

#if _MSC_VER == 1600
#    ifdef _M_X64
#        include <math.h>
#    endif // _M_X64
#endif //_MSC_VER == 1600

#if defined _WIN32 && _MSC_VER < 1900
//!\brief rename VC _isnan function as isnan for compatibility
//------------------------------------------------------------------------------
#    if _MSC_VER == 1800
#        error "Not tested with VS 2013"
#    endif // _MSC_VER == 1800
namespace std {
inline bool isnan(const double d) { return 0 != _isnan(d); }
} // namespace std
#endif // _WIN32 &&  _MSC_VER < 1900

#if defined _WIN32
//!\brief define constant M_PI for compatibility
#    define M_PI 3.14159265358979323846 /* pi */
#endif // _WIN32

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

constexpr double pi      = M_PI;
constexpr double piHalf  = pi / 2.0;
constexpr double twoPi   = 2.0 * pi;
constexpr double rad2deg = 180.0 / pi;
constexpr double deg2rad = pi / 180.0;

constexpr float pif      = static_cast<float>(pi);
constexpr float piHalff  = static_cast<float>(piHalf);
constexpr float twoPif   = static_cast<float>(twoPi);
constexpr float rad2degf = static_cast<float>(rad2deg);
constexpr float deg2radf = static_cast<float>(deg2rad);

constexpr float floatFromToQ4_12{4096};
//==============================================================================
//! \brief Template class to hold the powers for float/double fuzzy compare template functions.
//! \date June 18, 2018
//------------------------------------------------------------------------------
template<typename FloatType, uint8_t Exponent>
class NegativePower final
{
    static_assert(std::is_floating_point<FloatType>::value, "FloatType must be a floating-point type");
};

//definitions float
template<>
struct NegativePower<float, 0>
{
    constexpr float operator()() const { return 1e-0F; }
};
template<>
struct NegativePower<float, 1>
{
    constexpr float operator()() const { return 1e-1F; }
};
template<>
struct NegativePower<float, 2>
{
    constexpr float operator()() const { return 1e-2F; }
};
template<>
struct NegativePower<float, 3>
{
    constexpr float operator()() const { return 1e-3F; }
};
template<>
struct NegativePower<float, 4>
{
    constexpr float operator()() const { return 1e-4F; }
};
template<>
struct NegativePower<float, 5>
{
    constexpr float operator()() const { return 1e-5F; }
};
template<>
struct NegativePower<float, 6>
{
    constexpr float operator()() const { return 1e-6F; }
};
template<>
struct NegativePower<float, 7>
{
    constexpr float operator()() const { return 1e-7F; }
};
template<>
struct NegativePower<float, 8>
{
    constexpr float operator()() const { return 1e-8F; }
};
template<>
struct NegativePower<float, 9>
{
    constexpr float operator()() const { return 1e-9F; }
};
template<>
struct NegativePower<float, 10>
{
    constexpr float operator()() const { return 1e-10F; }
};
template<>
struct NegativePower<float, 11>
{
    constexpr float operator()() const { return 1e-11F; }
};
template<>
struct NegativePower<float, 12>
{
    constexpr float operator()() const { return 1e-12F; }
};
//definitions double
template<>
struct NegativePower<double, 0>
{
    constexpr double operator()() const { return 1e-0; }
};
template<>
struct NegativePower<double, 1>
{
    constexpr double operator()() const { return 1e-1; }
};
template<>
struct NegativePower<double, 2>
{
    constexpr double operator()() const { return 1e-2; }
};
template<>
struct NegativePower<double, 3>
{
    constexpr double operator()() const { return 1e-3; }
};
template<>
struct NegativePower<double, 4>
{
    constexpr double operator()() const { return 1e-4; }
};
template<>
struct NegativePower<double, 5>
{
    constexpr double operator()() const { return 1e-5; }
};
template<>
struct NegativePower<double, 6>
{
    constexpr double operator()() const { return 1e-6; }
};
template<>
struct NegativePower<double, 7>
{
    constexpr double operator()() const { return 1e-7; }
};
template<>
struct NegativePower<double, 8>
{
    constexpr double operator()() const { return 1e-8; }
};
template<>
struct NegativePower<double, 9>
{
    constexpr double operator()() const { return 1e-9; }
};
template<>
struct NegativePower<double, 10>
{
    constexpr double operator()() const { return 1e-10; }
};
template<>
struct NegativePower<double, 11>
{
    constexpr double operator()() const { return 1e-11; }
};
template<>
struct NegativePower<double, 12>
{
    constexpr double operator()() const { return 1e-12; }
};
template<>
struct NegativePower<double, 13>
{
    constexpr double operator()() const { return 1e-13; }
};
template<>
struct NegativePower<double, 14>
{
    constexpr double operator()() const { return 1e-14; }
};
template<>
struct NegativePower<double, 15>
{
    constexpr double operator()() const { return 1e-15; }
};
template<>
struct NegativePower<double, 16>
{
    constexpr double operator()() const { return 1e-16; }
};
template<>
struct NegativePower<double, 17>
{
    constexpr double operator()() const { return 1e-17; }
};

//==============================================================================
//!\brief Shortcut for the float NaN value.
//------------------------------------------------------------------------------
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

//==============================================================================
//!\brief Shortcut for the double NaN value.
//------------------------------------------------------------------------------
constexpr double NaN_double = std::numeric_limits<double>::quiet_NaN();

//==============================================================================
//!\brief Tests whether two \c float values are nearly equal.
//!\tparam EXP   The exponent of the epsilon used for the fuzzy
//!              compare. 10^(-EXP).
//!\param[in] a  First value to be compared with second value.
//!\param[in] b  Second value to be compared with first value.
//!\return \c True if the two \c float numbers are equal in
//!        terms of the machine precision, which means their
//!        difference must be less than 10^(-EXP).
//!
//!The exponent range is defined in NegFloatPotenciesOf10.
//------------------------------------------------------------------------------
template<uint8_t EXP>
inline bool fuzzyCompareT(const float a, const float b)
{
    return std::fabs(a - b) < NegativePower<float, EXP>{}();
}

//==============================================================================
//!\brief Tests whether two \c double values are nearly equal.
//!\tparam EXP   The exponent of the epsilon used for the fuzzy
//!              compare. 10^(-EXP).
//!\param[in] a  First value to be compared with second value.
//!\param[in] b  Second value to be compared with first value.
//!\return \c True if the two \c double numbers are equal in
//!        terms of the machine precision, which means their
//!        difference must be less than 10^(-EXP).
//!
//!The exponent range is defined in NegDoublePotenciesOf10.
//------------------------------------------------------------------------------
template<uint8_t EXP>
inline bool fuzzyCompareT(const double a, const double b)
{
    return std::abs(a - b) < NegativePower<double, EXP>{}();
}

//==============================================================================
//!\brief Fuzzy Compare two floats \a a and \a b. NaN equals NaN here.
//!\tparam EXP   The exponent of the epsilon used for the fuzzy
//!              compare. 10^(-EXP).
//!\param[in] a  First float to be compared.
//!\param[in] b  Second float to be compared.
//!\return \c True if the difference between \a a and \a b is not smaller
//!        than 10^(-EXP) or if both are NaN.
//!        \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t EXP>
inline bool fuzzyFloatEqualT(const float a, const float b)
{
    return ((std::isnan(a) && std::isnan(b)) || (std::isinf(a) && std::isinf(b))
            || (std::isfinite(a) && std::isfinite(b) && fuzzyCompareT<EXP>(a, b)));
}

//==============================================================================
//!\brief Fuzzy Compare two doubles \a a and \a b. NaN equals NaN here.
//!\tparam EXP   The exponent of the epsilon used for the fuzzy
//!              compare. 10^(-EXP).
//!\param[in] a  First double to be compared.
//!\param[in] b  Second double to be compared.
//!\return \c True if the difference between \a a and \a b is not smaller
//!        than 10^(-EXP) or if both are NaN.
//!        \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t EXP>
inline bool fuzzyDoubleEqualT(const double a, const double b)
{
    return ((std::isnan(a) && std::isnan(b)) || (std::isinf(a) && std::isinf(b))
            || (std::isfinite(a) && std::isfinite(b) && fuzzyCompareT<EXP>(a, b)));
}

//==============================================================================
//!\brief Fuzzy compare two float \a a and \a b. NaN equals NaN here.
//!\tparam EXP   The exponent of the epsilon used for fuzzy compare. 10^(-EXP).
//!\param[in] a  First float to be compared.
//!\param[in] b  Second float to be compared.
//!\return \c false if the difference between \a a and \a b is not smaller
//!        than 10^(-EXP) or if both are NaN.
//!        \c true otherwise.
//------------------------------------------------------------------------------
template<uint8_t EXP>
inline bool fuzzyFloatUnequalT(const float a, const float b)
{
    return (!fuzzyFloatEqualT<EXP>(a, b));
}

//==============================================================================
//!\brief Fuzzy compare two doubles \a a and \a b. NaN equals NaN here.
//!\tparam EXP   The exponent of the epsilon used for fuzzy compare. 10^(-EXP).
//!\param[in] a  First double to be compared.
//!\param[in] b  Second double to be compared.
//!\return \c false if the difference between \a a and \a b is not smaller
//!        than 10^(-EXP) or if both are NaN.
//!        \c true otherwise.
//------------------------------------------------------------------------------
template<uint8_t EXP>
inline bool fuzzyDoubleUnequalT(const double a, const double b)
{
    return (!fuzzyDoubleEqualT<EXP>(a, b));
}

//==============================================================================
//!\brief Normalize the given angle.
//!
//!Normalizes an angle given in radians by adding or subtracting an integer
//!multiple of 2*pi so that the resulting angle is in the half-open interval
//!(-pi,+pi]. The current implementation takes O(1) time, i.e. the time of
//!execution has a fixed upper boundary independent from the angle.
//!
//!\param[in] angleInRad  Angle to be normalized given in rad.
//!\return The normalized angle in (-pi, +pi].
//!\todo check whether (-pi, +pi] or [-pi, +pi) is correct.
//------------------------------------------------------------------------------
inline float normalizeRadians(float angleInRad)
{
    if (std::fabs(angleInRad) > (3.0F * pif))
    {
        // For numerical stability we must use this sin/cos/atan2
        // implementation even though it might consume more cycles.
        // Note that radians = -pi converts to atan2(0,-1) = +pi!
        angleInRad = std::atan2(std::sin(angleInRad), std::cos(angleInRad));
        // radians now in (-pi,+pi]
    } // if
    else
    {
        // fast version for "normal" out of range values
        while (angleInRad > pif)
        {
            angleInRad -= twoPif;
        } // while
        while (angleInRad <= -pif)
        {
            angleInRad += twoPif;
        } // while
    } // else
    return angleInRad;
}

//==============================================================================
//!\brief Normalize the given angle.
//!
//!Normalizes an angle given in radians by adding or subtracting an integer
//!multiple of 2*pi so that the resulting angle is in the half-open interval
//!(-pi,+pi]. The current implementation takes O(1) time, i.e. the time of
//!execution has a fixed upper boundary independent from the angle.
//!
//!\param[in] angleInRad  Angle to be normalized given in rad.
//!\return The normalized angle in (-pi, +pi].
//!\todo check whether (-pi, +pi] or [-pi, +pi) is correct.
//------------------------------------------------------------------------------
inline double normalizeRadians(double angleInRad)
{
    if (std::abs(angleInRad) > (3.0 * pi))
    {
        // For numerical stability we must use this sin/cos/atan2
        // implementation even though it might consume more cycles.
        // Note that radians = -pi converts to atan2(0,-1) = +pi!
        angleInRad = std::atan2(std::sin(angleInRad), std::cos(angleInRad));
        // radians now in (-pi,+pi]
    } // if
    else
    {
        // fast version for "normal" out of range values
        while (angleInRad > pi)
        {
            angleInRad -= twoPi;
        } // while
        while (angleInRad <= -pi)
        {
            angleInRad += twoPi;
        } // while
    } // else
    return angleInRad;
}

//==============================================================================
//!\brief Round to the closest integer.
//!
//!\param[in] floatValue The float value that shall be rounded.
//!\return \a floatValue rounded to the closest integer.
//------------------------------------------------------------------------------
template<typename IntT, typename = std::enable_if_t<std::is_integral<IntT>::value>>
inline constexpr IntT round(float floatValue) noexcept
{
    return IntT(floatValue + (floatValue >= 0.0F ? +0.5F : -0.5F));
}

//==============================================================================
//!\brief Round to the closest integer.
//!
//!\param[in] doubleValue The double value that shall be rounded.
//!\return \a doubleValue rounded to the closest integer.
//------------------------------------------------------------------------------
template<typename IntT, typename = std::enable_if_t<std::is_integral<IntT>::value>>
inline constexpr IntT round(double doubleValue) noexcept
{
    return IntT(doubleValue + (doubleValue >= 0.0 ? +0.5 : -0.5));
}

//==============================================================================
//!\brief Convert a value in a fixed comma notation Q4.12 to a float.
//!
//!\note In Q4.12 the first 4 bits describe the integer part and last 12 bits describe the fractional part.
//!
//!\param[in] valueAsQ4_12 The value in fixed comma notation Q4.12.
//!\return The value as float.
//------------------------------------------------------------------------------
constexpr inline float convertQ4_12ToFloat(uint16_t valueAsQ4_12) { return valueAsQ4_12 / floatFromToQ4_12; }

//==============================================================================
//!\brief Convert a value from float to a fixed comma notation Q4.12.
//!
//!\note In Q4.12 the first 4 bits describe the integer part and last 12 bits describe the fractional part.
//!
//!\param[in] valueAsFloat The value as float.
//!\return The value in fixed comma notation Q4.12.
//------------------------------------------------------------------------------
constexpr inline uint16_t convertFloatToQ4_12(float valueAsFloat)
{
    return static_cast<uint16_t>(valueAsFloat * floatFromToQ4_12);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
