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
//! \date Feb 20, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/ByteOrder.hpp>

#include <boost/predef/other/endian.h>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_fundamental.hpp>

#include <cstring>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a buffer.
//!
//! It is assumed that the stream provides the data in the same byte order
//! as the system the code is running on. So the byte order will left
//! untouched.
//!
//! \tparam T               Type of the value to be read.
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
void readLinear(const char*& srcBuf, T& value)
{
    memcpy(reinterpret_cast<char*>(&value), srcBuf, sizeof(value));
    srcBuf += sizeof(value);
}

//==============================================================================
//! \brief Read a value of type \a T from a buffer and flip the byte order.
//!
//! It is assumed that the stream provides the data in different byte order
//! as the system the code is running on. So the byte order has to be swapped.
//!
//! \tparam T               Type of the value to be read.
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
void readSwapped(const char*& srcBuf, T& value)
{
    readLinear(srcBuf, value);
    swapByteOrder(value);
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a buffer.
//!        Reading individual bytes is done in little-endian byte order.
//!
//! \tparam T               Type of the value to be read.
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
void readLE(const char*& srcBuf, T& value)
{
    readLinear(srcBuf, value);

#if BOOST_ENDIAN_BIG_BYTE
    swapByteOrder(value);
#else
    // keep byte order
#endif
}

//==============================================================================
//! \brief Read a value of type bool from a buffer.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<>
inline void readLE<bool>(const char*& srcBuf, bool& value)
{
    uint8_t tmp;
    readLinear(srcBuf, tmp); //  Endianness does not matter for 1 Byte read.
    value = (tmp != 0);
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a buffer. The expected byte order in
//!        the buffer is little-endian.
//!
//! \tparam T               Type of the value to be read.
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \return The value read from \a buffer. On big-endian machines the byte order
//!         has been swapped.
//------------------------------------------------------------------------------
template<typename T>
T readLE(const char*& buffer) noexcept
{
    static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                  "Cannot swap the byte order of non-fundamental or enum type T.");

    T value = *reinterpret_cast<const T*>(buffer); // read linear

#if BOOST_ENDIAN_BIG_BYTE
    swapByteOrder(value); // swap if machine has big endian architecture
#else
    //keep little endian byte order
#endif

    buffer += sizeof(T);
    return value;
}

//==============================================================================
//! \brief Read a value of type bool from a buffer.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \return \c true if the read value is \c true (i.e. not 0), \c false otherwise.
//!
//! \note readLE and readBE are identical for bool.
//------------------------------------------------------------------------------
template<>
inline bool readLE(const char*& buffer) noexcept
{
    const bool value = (*buffer != 0);
    ++buffer;
    return value;
}

//==============================================================================
//! \brief Read a 24 bit value from a buffer into an uint32_t \a value.
//!        The 3 bytes of the input are expected to be in little-endian byte order.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On exit \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readLE24(const char*& srcBuf, uint32_t& value)
{
    memcpy(reinterpret_cast<char*>(&value), srcBuf, 3);
    srcBuf += 3;

#if BOOST_ENDIAN_BIG_BYTE
    swapByteOrder(value);
#else
    // nothing to do
#endif

    // clear the most significant byte
    value = value & 0x00FFFFFFU;
}

//==============================================================================
//! \brief Read a 24 bit value from a buffer into an int32_t \a value.
//!        The 3 bytes of the input are expected to be in little-endian byte order.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On exit \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readLE24(const char*& srcBuf, int32_t& value)
{
    uint32_t& asUnsigned = *reinterpret_cast<uint32_t*>(&value);
    readLE24(srcBuf, asUnsigned);

    // most significant bit (here bit 23) set indicates negative number in 2-complement.
    if ((asUnsigned & 0x00800000U) == 0x00800000U)
    {
        // clear the most significant byte in 2-complement.
        asUnsigned = asUnsigned | (0xFFU << 24);
    }
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a buffer.
//!        Reading individual bytes is done in big-endian byte order.
//!
//! \tparam T               Type of the value to be read.
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
void readBE(const char*& srcBuf, T& value)
{
    readLinear(srcBuf, value);

#if BOOST_ENDIAN_LITTLE_BYTE
    swapByteOrder(value);
#else
    // keep byte order
#endif
}

//==============================================================================
//! \brief Read a value of type bool from a buffer.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<>
inline void readBE<bool>(const char*& srcBuf, bool& value)
{
    uint8_t tmp;
    readLinear(srcBuf, tmp); // Endianness does not matter for 1 Byte read.
    value = (tmp != 0);
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a buffer. The expected byte order in
//!        the buffer is big-endian.
//!
//! \tparam T               Type of the value to be read.
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \return The value read from \a buffer. On little-endian machines the byte order
//!         has been swapped.
//------------------------------------------------------------------------------
template<typename T>
T readBE(const char*& buffer) noexcept
{
    static_assert((std::is_fundamental<T>::value) || std::is_enum<T>::value,
                  "Cannot swap the byte order of non-fundamental or enum type T.");

    T value = *reinterpret_cast<const T*>(buffer); // read linear

#if BOOST_ENDIAN_LITTLE_BYTE
    swapByteOrder(value); // swap if machine has little endian architecture
#else
    // keep big endian byte order
#endif

    buffer += sizeof(T);
    return value;
}

//==============================================================================
//! \brief Read a value of type bool from a buffer.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \return \c true if the read value is \c true (i.e. not 0), \c false otherwise.
//!
//! \note readBE and readLE are identical for bool.
//------------------------------------------------------------------------------
template<>
inline bool readBE(const char*& buffer) noexcept
{
    const bool value = (*buffer != 0);
    ++buffer;
    return value;
}

//==============================================================================
//! \brief Read a 24 bit value from a buffer into an uint32_t \a value.
//!        The 3 bytes of the input are expected to be in big-endian byte order.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On exit \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readBE24(const char*& srcBuf, uint32_t& value)
{
    // Always read into target value as if the machine is big endian.
    // The first target byte (most significant byte in big endian) will
    // not be overridden and is still unchanged. It will set later to 0.
    memcpy(reinterpret_cast<char*>(&value) + 1, srcBuf, 3);
    srcBuf += 3;

#if BOOST_ENDIAN_LITTLE_BYTE
    swapByteOrder(value);
#else
    // nothing to do
#endif

    // clear the most significant byte (independent of endianess)
    value = value & 0x00FFFFFFU;
}

//==============================================================================
//! \brief Read a 24 bit value from a buffer into an int32_t \a value.
//!        The 3 bytes of the input are expected to be in big-endian byte order.
//!
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On exit \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readBE24(const char*& srcBuf, int32_t& value)
{
    uint32_t& asUnsigned = *reinterpret_cast<uint32_t*>(&value);
    readBE24(srcBuf, asUnsigned);

    // most significant bit (here bit 23) set indicates negative number in 2-complement.
    if ((asUnsigned & 0x00800000U) == 0x00800000U)
    {
        // set the most significant byte in 2-complement for negative numbers.
        asUnsigned = asUnsigned | (0xFFU << 24);
    }
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a buffer.
//!
//! \tparam T               Type of the value to be read.
//! \param[in, out] srcBuf  Buffer providing the data to be read.
//!                         On output \a srcBuf has been increased
//!                         by the number of bytes that have been read.
//! \param[out]     value   On exit it will hold the value that has been read.
//! \param[in]      sourceIsBigEndian
//!                        Set to \c true, if the stream has big-endian byte order
//!                        or to \c false, if it has little-endian byte order.
//------------------------------------------------------------------------------
template<typename T>
void read(const char*& srcBuf, T& value, bool sourceIsBigEndian = false)
{
#if BOOST_ENDIAN_BIG_BYTE
    const bool do_swap = !sourceIsBigEndian;
#else
    const bool do_swap = sourceIsBigEndian;
#endif

    if (do_swap)
        readSwapped(srcBuf, value);
    else
        readLinear(srcBuf, value);
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Write a value of type \a T into a buffer.
//!
//! It is assumed that the stream receiving the data in the same byte order
//! as the system the code is running on. So the byte order will left
//! untouched.
//!
//! \tparam T                  Type of the value to be written.
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeLinear(char*& targetBuf, const T& value)
{
    memcpy(targetBuf, reinterpret_cast<const char*>(&value), sizeof(value));
    targetBuf += sizeof(value);
}

//==============================================================================
//! \brief Write a value of type \a T into a buffer.
//!
//! It is assumed that the stream receiving the data in different byte order
//! as the system the code is running on. So the byte order  has to be swapped.
//!
//! \tparam T                  Type of the value to be written.
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeSwapped(char*& targetBuf, const T& value)
{
    const T swappedValue = getWithSwappedByteOrder(value);

    T* targetValue = reinterpret_cast<T*>(targetBuf);
    *targetValue   = swappedValue;

    targetBuf += sizeof(value);
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Write a value of type \a T into a buffer.
//!        Writing individual bytes is done in little-endian byte order.
//!
//! \tparam T                  Type of the value to be written.
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeLE(char*& targetBuf, const T& value)
{
#if BOOST_ENDIAN_BIG_BYTE
    writeSwapped(targetBuf, value);
#else
    writeLinear(targetBuf, value);
#endif
}

//==============================================================================
//! \brief Write a bool value into a buffer.
//!
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<>
inline void writeLE<bool>(char*& targetBuf, const bool& value)
{
    const uint8_t tmp = value ? 1 : 0;
    writeLinear(targetBuf, tmp); // Endianness does not matter for 1 Byte read.
}

//==============================================================================
//! \brief Write a 24 bit value from a variable of type \a T into a buffer.
//!        The 3 bytes are written in little-endian byte order to the output.
//!
//! \tparam T                  Type of the value to be written.
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeLE24(char*& targetBuf, const T& value)
{
    static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
                  "Writing 24 bit values is implemented for types uint32_t or int32_t only!");

#if BOOST_ENDIAN_LITTLE_BYTE
    memcpy(targetBuf, reinterpret_cast<const char*>(&value), 3); // ignore last, most significant byte.
#else
    memcpy(targetBuf, reinterpret_cast<const char*>(&value) + 1, 3); // ingore first, most significant byte.
    std::swap(*targetBuf, *(targetBuf + 2)); // swap the first and third byte.
#endif
    targetBuf += 3;
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Write a value of type \a T into a buffer.
//!        Writing individual bytes is done in big-endian byte order.
//!
//! \tparam T                  Type of the value to be written.
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeBE(char*& targetBuf, const T& value)
{
#if BOOST_ENDIAN_BIG_BYTE
    writeLinear(targetBuf, value);
#else
    writeSwapped(targetBuf, value);
#endif
}

//==============================================================================
//! \brief Write a bool value into a buffer.
//!
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<>
inline void writeBE(char*& targetBuf, const bool& value)
{
    const uint8_t tmp = value ? 1 : 0;
    writeLinear(targetBuf, tmp); // Endianness does not matter for 1 Byte read.
}

//==============================================================================
//! \brief Write a 24 bit value from a variable of type \a T into a buffer.
//!        The 3 bytes are written in big-endian byte order to the output.
//!
//! \tparam T                  Type of the value to be written.
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeBE24(char*& targetBuf, const T& value)
{
    static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
                  "Writing 24 bit values is implemented for types uint32_t or int32_t only!");

#if BOOST_ENDIAN_LITTLE_BYTE
    memcpy(targetBuf, reinterpret_cast<const char*>(&value), 3); // ignore last, most significant byte.
    std::swap(*targetBuf, *(targetBuf + 2)); // swap the first and third byte.
#else
    memcpy(targetBuf, reinterpret_cast<const char*>(&value) + 1, 3); // ingore first, most significant byte.
#endif
    targetBuf += 3;
}
//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Write a value of type \a T into a buffer.
//!
//! \tparam T                  Type of the value to be written.
//! \param[in, out] targetBuf  Buffer receiving the data been written.
//!                            On output \a targetBuf has been increased
//!                            by the number of bytes that has been written.
//! \param[in]      value      The value to be written.
//! \param[in]      destIsBigEndian
//!                            Set to \c true, if the stream has big-endian byte order
//!                            or to \c false, if it has little-endian byte order.
//------------------------------------------------------------------------------
template<typename T>
void write(char*& targetBuf, const T& value, bool destIsBigEndian = false)
{
#if BOOST_ENDIAN_BIG_BYTE
    const bool do_swap = !destIsBigEndian;
#else
    const bool do_swap = destIsBigEndian;
#endif

    if (do_swap)
        writeSwapped(targetBuf, value);
    else
        writeLinear(targetBuf, value);
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
