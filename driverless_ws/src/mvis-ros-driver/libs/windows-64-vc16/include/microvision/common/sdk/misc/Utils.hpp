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
//! \date Jun 5, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <type_traits>
#include <memory>
#include <string>
#include <locale>

//==============================================================================
namespace std {
//==============================================================================

#if SDK_CPPSTD_VERSION <= 201703L // Enable if lower than c++20 version.

//==============================================================================
//! \brief Removes const and/or volatile and reference from the given type.
//------------------------------------------------------------------------------
template<typename ValueType>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<ValueType>::type>::type;

#endif // if SDK_CPPSTD_VERSION <= 201703L

//==============================================================================
} // namespace std
//==============================================================================

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Cast a unique_ptr<T> dynamic to another type of the inherit tree of T.
//! \tparam ToType      Any kind of type in inherit tree of FromType.
//! \tparam FromType    Any kind of type in inherit tree of ToType.
//! \param[in] ptr      Unique pointer which contains an instance of type FromType as move value.
//! \return Unique pointer which contains either an instance of type ToType is castable or otherwise nullptr.
//------------------------------------------------------------------------------
template<typename ToType, typename FromType>
inline std::unique_ptr<ToType> dynamic_pointer_cast(std::unique_ptr<FromType>&& ptr)
{
    ToType* testPtr = dynamic_cast<ToType*>(ptr.get());

    if (testPtr == nullptr)
    {
        ptr.reset();
        return nullptr;
    }
    else
    {
        return std::unique_ptr<ToType>(dynamic_cast<ToType*>(ptr.release()));
    }
}

//==============================================================================
//! \brief Compared the start of a string.
//! \tparam TCharL          Any type of "char" for the left string.
//! \tparam TCharR          Any type of "char" for the right string.
//! \param[in] strL         Left string to compare.
//! \param[in] strR         Right string with what strL has to start.
//! \param[in] ignoreCase   Ignore case for compare (Optional, default=false).
//! \param[in] loc          Locale for char normalization (Optional, default=system).
//! \return Either \c true if the strL starts with strR, otherwise \c false.
//------------------------------------------------------------------------------
template<typename TCharL, typename TCharR>
bool startsWith(const std::basic_string<TCharL>& strL,
                const std::basic_string<TCharR>& strR,
                bool ignoreCase = false,
                std::locale loc = std::locale{})
{
    using lsize_t = typename std::basic_string<TCharL>::size_type;
    using rsize_t = typename std::basic_string<TCharR>::size_type;

    if (strL.length() >= strR.length())
    {
        for (rsize_t ri = 0; ri < strR.length(); ++ri)
        {
            const lsize_t li = static_cast<lsize_t>(ri);

            if (ignoreCase)
            {
                if (std::tolower(strL[li], loc) != std::tolower(strR[ri], loc))
                {
                    return false;
                }
            }
            else
            {
                if (strL[li] != strR[ri])
                {
                    return false;
                }
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}

//==============================================================================
//! \brief Compared the end of a string.
//! \tparam TCharL          Any type of "char" for the left string.
//! \tparam TCharR          Any type of "char" for the right string.
//! \param[in] strL         Left string to compare.
//! \param[in] strR         Right string with what strL has to end.
//! \param[in] ignoreCase   Ignore case for compare (Optional, default=false).
//! \param[in] loc          Locale for char normalization (Optional, default=system).
//! \return Either \c true if the strL ends with strR, otherwise \c false.
//------------------------------------------------------------------------------
template<typename TCharL, typename TCharR>
bool endsWith(const std::basic_string<TCharL>& strL,
              const std::basic_string<TCharR>& strR,
              bool ignoreCase = false,
              std::locale loc = std::locale{})
{
    using lsize_t = typename std::basic_string<TCharL>::size_type;
    using rsize_t = typename std::basic_string<TCharR>::size_type;

    if (strL.length() >= strR.length())
    {
        rsize_t rIndex{0};
        const lsize_t lStartIndex = static_cast<lsize_t>(strL.length() - strR.length());

        for (lsize_t li = lStartIndex; li < strL.length(); ++li)
        {
            if (ignoreCase)
            {
                if (std::tolower(strL[li], loc) != std::tolower(strR[rIndex], loc))
                {
                    return false;
                }
            }
            else
            {
                if (strL[li] != strR[rIndex])
                {
                    return false;
                }
            }

            ++rIndex;
        }
    }
    else
    {
        return false;
    }

    return true;
}

//==============================================================================
//! \brief Compares two strings.
//! \tparam TCharL          Any type of "char" for the left string.
//! \tparam TCharR          Any type of "char" for the right string.
//! \param[in] strL         Left string to compare.
//! \param[in] strR         Right string to compare.
//! \param[in] ignoreCase   Ignore case for compare (Optional, default=false).
//! \param[in] loc          Locale for char normalization (Optional, default=system).
//! \return Either \c true if the strL and strR are equal, otherwise \c false.
//------------------------------------------------------------------------------
template<typename TCharL, typename TCharR>
bool compare(const std::basic_string<TCharL>& strL,
             const std::basic_string<TCharR>& strR,
             bool ignoreCase = false,
             std::locale loc = std::locale{})
{
    using lsize_t = typename std::basic_string<TCharL>::size_type;
    using rsize_t = typename std::basic_string<TCharR>::size_type;

    if (strL.length() == strR.length())
    {
        for (rsize_t ri = 0; ri < strR.length(); ++ri)
        {
            const lsize_t li = static_cast<lsize_t>(ri);

            if (ignoreCase)
            {
                if (std::tolower(strL[li], loc) != std::tolower(strR[ri], loc))
                {
                    return false;
                }
            }
            else
            {
                if (strL[li] != strR[ri])
                {
                    return false;
                }
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}

//==============================================================================
//! \brief Get value if type is same
//! \tparam FromType    Value type
//! \tparam ToType      Required value type
//! \param[in] value    Value of type \c FromType
//! \return Value of type \c ToType
//------------------------------------------------------------------------------
template<typename FromType,
         typename ToType,
         typename std::enable_if<std::is_same<FromType, ToType>::value, int>::type = 0>
inline ToType staticCastIfNeeded(const FromType value)
{
    return value;
}

//==============================================================================
//! \brief Get static casted value if type is not same
//! \tparam FromType    Value type
//! \tparam ToType      Required value type
//! \param[in] value    Value of type \c FromType
//! \return Value of type \c ToType
//------------------------------------------------------------------------------
template<typename FromType,
         typename ToType,
         typename std::enable_if<!std::is_same<FromType, ToType>::value, int>::type = 0>
inline ToType staticCastIfNeeded(const FromType value)
{
    return static_cast<ToType>(value);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
