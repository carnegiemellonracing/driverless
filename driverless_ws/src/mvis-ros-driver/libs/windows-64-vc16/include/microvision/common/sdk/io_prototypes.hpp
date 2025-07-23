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
//! \date Apr 24, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <iostream>
#include <type_traits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<typename T>
void writeLE(std::ostream& os, const T& value);
template<typename T>
void writeBE(std::ostream& os, const T& value);

template<typename T>
void readLE(std::istream& is, T& value);
template<typename T>
void readBE(std::istream& is, T& value);
template<typename T>
T readBE(std::istream& is);
template<typename T>
T readLE(std::istream& is);

// default implementation
template<typename T>
inline constexpr std::streamsize serializedSize(const T&)
{
    static_assert(std::is_fundamental<T>::value,
                  "Called serializedSize template function with complex type. No specialization available");

    return std::streamsize(sizeof(T));
}

//==============================================================================

template<typename ENUMTYPE, typename INTTYPE>
void writeBE(std::ostream& os, const ENUMTYPE& value)
{
    writeBE(os, INTTYPE(value));
}

//==============================================================================

template<typename ENUMTYPE, typename INTTYPE>
void readBE(std::istream& is, ENUMTYPE& value)
{
    INTTYPE s;
    readBE(is, s);
    value = ENUMTYPE(s);
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
