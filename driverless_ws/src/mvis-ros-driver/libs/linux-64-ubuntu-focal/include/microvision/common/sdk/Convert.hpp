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
//! \date Jun 1, 2012
//------------------------------------------------------------------------------

#pragma once
//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <ios>

//#include <arpa/inet.h>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Convert the given value v into a string.
//! \tparam T     Type of the object to be converted into a string.
//!               There must exist an operator<< for type T.
//! \param[in] v  Value to be converted into a string.
//! \return The string representation of \a v.
//------------------------------------------------------------------------------
template<typename T>
inline std::string toString(const T& v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

//==============================================================================
//! \brief Specialization of toString for a char.
//! \param[in] v  Value to be converted into a string.
//!               The value shall be shown as decimal number
//!               hence it will be converted into an integer
//!               before written into the string.
//! \return The string representation of \a v.
//------------------------------------------------------------------------------
template<>
inline std::string toString<char>(const char& v)
{
    std::stringstream ss;
    ss << int(v);
    return ss.str();
}

//==============================================================================
//! \brief Specialization of toString for a unsigned char.
//! \param[in] v  Value to be converted into a string.
//!               The value shall be shown as decimal number
//!               hence it will be converted into an integer
//!               before written into the string.
//! \return The string representation of \a v.
//------------------------------------------------------------------------------
template<>
inline std::string toString<unsigned char>(const unsigned char& v)
{
    std::stringstream ss;
    ss << int(v);
    return ss.str();
}

//==============================================================================
//! \brief Convert the given value vector into a string.
//!
//! This method read the text and convert the in text form given
//! values into a vector of the given type \a T.
//!
//! \tparam T      Type of the vector entries to be read from the string \a s.
//!                There must exist an operator>> for type T.
//! \param[in]  s  String to be read into a vector.
//! \param[out] v  On exit this vector will hold the read values.
//------------------------------------------------------------------------------
template<typename T>
inline void fromString2Vector(const std::string& s, std::vector<T>& v)
{
    v.clear();
    if (s.empty())
        return;

    std::stringstream ss(s);
    T e;

    while (!ss.eof())
    {
        ss >> e;
        v.push_back(e);
    }
}
//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
