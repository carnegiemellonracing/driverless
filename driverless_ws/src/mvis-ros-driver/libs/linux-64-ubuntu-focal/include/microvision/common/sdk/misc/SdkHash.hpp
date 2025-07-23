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
//! \date Dec 15, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <string>

#include <cstdint>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FNV-1a hash for 64bit length
//!
//! \param[in] classId  Null terminated C string.
//! \note The computation actually visits the string from back to front and not
//!       from front to back.
//------------------------------------------------------------------------------
constexpr inline uint64_t hash(const char* classId)
{
    return (*classId ? (hash(classId + 1) ^ static_cast<unsigned char>(*classId)) * 0x100000001b3UL
                     : 0xcbf29ce484222325UL);
}

//==============================================================================

//==============================================================================
//! \brief FNV-1a hash for 64bit length
//!
//! \param[in] start  The start of the string.
//! \param[in] end    Beyond the end of the string.
//! \note The computation actually visits the string from back to front and not
//!       from front to back.
//------------------------------------------------------------------------------
inline uint64_t hash(std::string::const_iterator start, std::string::const_iterator end)
{
    return ((start != end) ? (hash(start + 1, end) ^ static_cast<unsigned char>(*start)) * 0x100000001b3UL
                           : 0xcbf29ce484222325UL);
}

//==============================================================================
//! \brief Helper structure for getting the hash value of an enum class.
//------------------------------------------------------------------------------
struct EnumClassHash
{
    template<typename T>
    uint64_t operator()(T t) const
    {
        return static_cast<uint64_t>(t);
    }
};

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
