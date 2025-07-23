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
//! \date Mar 8, 2024
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <cstdint>
#include <utility>
#include <type_traits>
#include <cstring>

//==============================================================================

static_assert(sizeof(char) == 1, "Size of type char has to be equal to 1 byte.");

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Template class that provides the means to swap the byte order of any
//!        fundamental or enum (with underlying integral, i.e. fundamental) type.
//! \tparam typeSize  The size of the the type whose byte order shall be swapped.
//!                   (Use to allow specializations).
//! \note The construct template-struct + (static) template-member-functions has
//!       been chosen since partial specialization of function templates is not
//!       possible until C++20.
//------------------------------------------------------------------------------
template<size_t typeSize>
struct ByteOrder
{
#ifndef _WIN32
#    pragma GCC diagnostic ignored "-Warray-bounds"
#endif // _WIN32

    //========================================
    //! \brief Generic function to swap the byte order of any  fundamental or enum
    //!        (with underlying integral, i.e. fundamental) type \a T of size
    //!        \a typeSize.
    //! \tparam T  The type of the \a value whose byte order shall be swapped.
    //! \param[in,out] value  On exit the byte order has been swapped with
    //!                       respect of the byte order on entry.
    //----------------------------------------
    template<typename T>
    static void swap(T& value) noexcept
    {
        static_assert(sizeof(T) == typeSize, "Type and size mismatch.");
        static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                      "Cannot swap the byte order of non-fundamental or enum type T.");

        const size_t size = sizeof(value);

        char* const data = reinterpret_cast<char*>(&value);
        char* incDataPtr = data;
        char* decDataPtr = data + size - 1;

        while (incDataPtr < decDataPtr)
        {
            std::swap(*incDataPtr++, *decDataPtr--);
        }
    }

#ifndef _WIN32
#    pragma GCC diagnostic warning "-Warray-bounds"
#endif // _WIN32
};

//==============================================================================

//==============================================================================
//! \brief Template class that provides the means to swap the byte order of a
//!        fundamental or enum (with underlying integral, i.e. fundamental) type
//!        of size 1. Size 1 means that there isn't anything to swap.
//------------------------------------------------------------------------------
template<>
struct ByteOrder<1>
{
    //========================================
    //! \brief Swap the byte order of a type \a T of size 1. This function will
    //!        do nothing.
    //! \attention It is recommended not to use this function directly but
    //!            swapByteOrder instead.
    //----------------------------------------
    template<typename T>
    static void swap(T&) noexcept
    {
        static_assert(sizeof(T) == 1, "Type and size mismatch.");
        // nothing to swap
    }
};

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Specialization to provide the means to swap the byte order of any
//!        fundamental or enum (with underlying integral, i.e. fundamental) type
//!        of size 2.
//------------------------------------------------------------------------------
template<>
struct ByteOrder<2>
{
    //========================================
    //! \brief Swap the byte order of a \a value of type \a T of size 2.
    //! \tparam T  The type of the \a value whose byte order shall be swapped.
    //! \param[in,out] value  On exit the byte order has been swapped with
    //!                       respect of the byte order on entry.
    //! \attention It is recommended not to use this function directly but
    //!            swapByteOrder instead.
    //!----------------------------------------
    template<typename T>
    static void swap(T& value) noexcept;
};

//==============================================================================
//! \brief Swap the byte order of an uint16_t \a value.
//! \param[in,out] value  On exit the byte order has been swapped with
//!                       respect of the byte order on entry.
//! \attention It is recommended not to use this function directly but
//!            swapByteOrder instead.
//------------------------------------------------------------------------------
template<>
inline void ByteOrder<2>::swap(uint16_t& value) noexcept
{
    value = static_cast<uint16_t>(static_cast<uint16_t>(value >> 8) & 0x00FFU) //
            | static_cast<uint16_t>(static_cast<uint16_t>(value << 8) & 0xFF00U);
}

//==============================================================================
//! \brief Swap the byte order of a \a value of type \a T of size 2.
//! \tparam T  The type of the \a value whose byte order shall be swapped.
//! \param[in,out] value  On exit the byte order has been swapped with
//!                       respect of the byte order on entry.
//! \attention It is recommended not to use this function directly but
//!            swapByteOrder instead.
//------------------------------------------------------------------------------
template<typename T>
void ByteOrder<2>::swap(T& value) noexcept
{
    static_assert(sizeof(T) == 2, "Type and size mismatch.");
    static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                  "Cannot swap the byte order of non-fundamental or enum type T.");

    uint16_t u16;
    memcpy(&u16, &value, sizeof(T));
    swap(u16); // fall back to conversion of an uint16_t
    memcpy(&value, &u16, sizeof(T));
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Specialization to provide the means to swap the byte order of any
//!        fundamental or enum (with underlying integral, i.e. fundamental) type
//!        of size 4.
//------------------------------------------------------------------------------
template<>
struct ByteOrder<4>
{
    //========================================
    //! \brief Swap the byte order of a \a value of type \a T of size 4.
    //! \tparam T  The type of the \a value whose byte order shall be swapped.
    //! \param[in,out] value  On exit the byte order has been swapped with
    //!                       respect of the byte order on entry.
    //! \attention It is recommended not to use this function directly but
    //!            swapByteOrder instead.
    //!----------------------------------------
    template<typename T>
    static void swap(T& value) noexcept;
};

//==============================================================================
//! \brief Swap the byte order of an uint32_t \a value.
//! \param[in,out] value  On exit the byte order has been swapped with
//!                       respect of the byte order on entry.
//! \attention It is recommended not to use this function directly but
//!            swapByteOrder instead.
//!-----------------------------------------------------------------------------
template<>
inline void ByteOrder<4>::swap(uint32_t& value) noexcept
{
    value = static_cast<uint32_t>(static_cast<uint32_t>(value >> 24) & 0x000000FFUl) //
            | static_cast<uint32_t>(static_cast<uint32_t>(value >> 8) & 0x0000FF00U) //
            | static_cast<uint32_t>(static_cast<uint32_t>(value << 8) & 0x00FF0000U) //
            | static_cast<uint32_t>(static_cast<uint32_t>(value << 24) & 0xFF000000U);
}

//==============================================================================
//! \brief Swap the byte order of a \a value of type \a T of size 4.
//! \tparam T  The type of the \a value whose byte order shall be swapped.
//! \param[in,out] value  On exit the byte order has been swapped with
//!                       respect of the byte order on entry.
//! \attention It is recommended not to use this function directly but
//!            swapByteOrder instead.
//------------------------------------------------------------------------------
template<typename T>
void ByteOrder<4>::swap(T& value) noexcept
{
    static_assert(sizeof(T) == 4, "Type and size mismatch.");
    static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                  "Cannot swap the byte order of non-fundamental or enum type T.");

    uint32_t u32;
    memcpy(&u32, &value, sizeof(T));
    swap(u32); // fall back to conversion of an uint32_t
    memcpy(&value, &u32, sizeof(T));
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Specialization to provide the means to swap the byte order of any
//!        fundamental or enum (with underlying integral, i.e. fundamental) type
//!        of size 8.
//------------------------------------------------------------------------------
template<>
struct ByteOrder<8>
{
    //========================================
    //! \brief Swap the byte order of a \a value of type \a T of size 8.
    //! \tparam T  The type of the \a value whose byte order shall be swapped.
    //! \param[in,out] value  On exit the byte order has been swapped with
    //!                       respect of the byte order on entry.
    //! \attention It is recommended not to use this function directly but
    //!            swapByteOrder instead.
    //!----------------------------------------
    template<typename T>
    static void swap(T& value) noexcept;
};

//==============================================================================
//! \brief Swap the byte order of an uint64_t \a value.
//! \param[in,out] value  On exit the byte order has been swapped with
//!                       respect of the byte order on entry.
//! \attention It is recommended not to use this function directly but
//!            swapByteOrder instead.
//!-----------------------------------------------------------------------------
template<>
inline void ByteOrder<8>::swap(uint64_t& value) noexcept
{
    value = static_cast<uint64_t>(static_cast<uint64_t>(value >> 56) & 0x00000000000000FFU) //
            | static_cast<uint64_t>(static_cast<uint64_t>(value >> 40) & 0x000000000000FF00U) //
            | static_cast<uint64_t>(static_cast<uint64_t>(value >> 24) & 0x0000000000FF0000U) //
            | static_cast<uint64_t>(static_cast<uint64_t>(value >> 8) & 0x00000000FF000000U) //
            | static_cast<uint64_t>(static_cast<uint64_t>(value << 8) & 0x000000FF00000000U) //
            | static_cast<uint64_t>(static_cast<uint64_t>(value << 24) & 0x0000FF0000000000U) //
            | static_cast<uint64_t>(static_cast<uint64_t>(value << 40) & 0x00FF000000000000U) //
            | static_cast<uint64_t>(static_cast<uint64_t>(value << 56) & 0xff00000000000000U);
}

//==============================================================================
//! \brief Swap the byte order of a \a value of type \a T of size 8.
//! \tparam T  The type of the \a value whose byte order shall be swapped.
//! \param[in,out] value  On exit the byte order has been swapped with
//!                       respect of the byte order on entry.
//! \attention It is recommended not to use this function directly but
//!            swapByteOrder instead.
//!-----------------------------------------------------------------------------
template<typename T>
void ByteOrder<8>::swap(T& value) noexcept
{
    static_assert(sizeof(T) == 8, "Type and size mismatch.");
    static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                  "Cannot swap the byte order of non-fundamental or enum type T.");

    uint64_t u64;
    memcpy(&u64, &value, sizeof(T));
    swap(u64); // fall back to conversion of an uint64_t
    memcpy(&value, &u64, sizeof(T));
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Swap the byte order of a value of any fundamental or enum
//!       (with underlying integral, i.e. fundamental) type.
//! \tparam T  The type of the \a value whose byte order shall be swapped.
//! \param[in,out] value  On exit the byte order has been swapped with
//!                       respect of the byte order on entry.
//------------------------------------------------------------------------------
template<typename T>
void swapByteOrder(T& value) noexcept
{
    static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                  "Cannot swap the byte order of non-fundamental or enum type T.");

    ByteOrder<sizeof(T)>::swap(value);
}

//==============================================================================
//! \brief Swap the byte order of a value of any fundamental or enum
//!       (with underlying integral, i.e. fundamental) type.
//! \tparam T  The type of the \a value whose byte order shall be swapped.
//! \param[in] value  The value of type \a T whose byte order shall be swapped.
//! \return The \a value with swapped byte order.
//------------------------------------------------------------------------------
template<typename T>
T getWithSwappedByteOrder(const T value) noexcept
{
    static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                  "Cannot swap the byte order of non-fundamental or enum type T.");

    T swappedValue = value;
    ByteOrder<sizeof(T)>::swap(swappedValue);

    return swappedValue;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
