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
//! \date Sep 4, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/io_prototypes.hpp>
#include <boost/predef/other/endian.h>
#include <microvision/common/sdk/ByteOrder.hpp>

#include <iostream>
#include <type_traits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a stream.
//!
//! It is assumed that the stream provides the data in the same byte order
//! as the system the code is running on. So the byte order will be left
//! untouched.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
inline void readLinear(std::istream& is, T& value)
{
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
}

//==============================================================================
//! \brief Read a value of type \a T for stream \a is and flip the byte order.
//!
//! It is assumed that the stream provides the data in different byte order
//! as the system the code is running on. So the byte order has to be swapped.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
void readSwapped(std::istream& is, T& value)
{
    readLinear(is, value);
    swapByteOrder(value);
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Read a value of type \a T from a stream.
//!        Reading individual bytes is done in little-endian byte order.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
inline void readLE(std::istream& is, T& value)
{
    static_assert(std::is_fundamental<T>::value,
                  "Called readLE template function with complex type. No specialization available");

    readLinear(is, value);

#if BOOST_ENDIAN_BIG_BYTE
    swapByteOrder(value);
#else
    // keep byte order
#endif
}

//==============================================================================
//! \brief Read a value of type \a T from a stream.
//!        Reading individual bytes is done in little-endian byte order.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \return The value that has been read.
//------------------------------------------------------------------------------
template<typename T>
inline T readLE(std::istream& is)
{
    static_assert(std::is_fundamental<T>::value,
                  "Called readBE template function with complex type. No specialization available");
    T value = T();
    readLinear(is, value);

#if BOOST_ENDIAN_BIG_BYTE
    swapByteOrder(value);
#else
    // keep byte order
#endif

    return value;
}

//==============================================================================
//! \brief Read a value of type bool from a stream.
//!
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<>
inline void readLE<bool>(std::istream& is, bool& value)
{
    uint8_t tmp;
    readLinear(is, tmp); //  Endianness does not matter for 1 Byte read.
    value = (tmp != 0);
}

//==============================================================================
//! \brief Read a 24 bit value from a stream into the uint32_t \a value.
//!        The 3 bytes are read in little-endian byte order from the input.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readLE24(std::istream& is, uint32_t& value)
{
    char* targetBuf = reinterpret_cast<char*>(&value);
    is.read(targetBuf, 3);

#if BOOST_ENDIAN_BIG_BYTE
    swapByteOrder(value);
#else
    // nothing to do
#endif

    // clear the most significant byte
    value = value & 0x00FFFFFFU;
}

//==============================================================================
//! \brief Read a 24 bit value from a stream into the int32_t \a value.
//!        The 3 bytes are read in little-endian byte order from the input.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readLE24(std::istream& is, int32_t& value)
{
    uint32_t& asUnsigned = *reinterpret_cast<uint32_t*>(&value);
    readLE24(is, asUnsigned);

    // most significant bit (here bit 23) set indicates negative number in 2-complement.
    if ((asUnsigned & 0x00800000U) == 0x00800000U)
    {
        // clear the most significant byte in 2-complement.
        asUnsigned = asUnsigned | (0xFFU << 24U);
    }
}

//==============================================================================
//! \brief Read a value of type \a T from a stream.
//!        Reading individual bytes is done in big-endian byte order.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \return The value that has been read.
//------------------------------------------------------------------------------
template<typename T>
inline T readBE(std::istream& is)
{
    static_assert(std::is_fundamental<T>::value,
                  "Called readBE template function with complex type. No specialization available");
    T value = T();
    readLinear(is, value);

#if BOOST_ENDIAN_LITTLE_BYTE
    swapByteOrder(value);
#else
    // keep byte order
#endif

    return value;
}

//==============================================================================
//! \brief Read a value of type \a T from a stream.
//!        Reading individual bytes is done in big-endian byte order.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<typename T>
inline void readBE(std::istream& is, T& value)
{
    static_assert(std::is_fundamental<T>::value,
                  "Called readBE template function with complex type. No specialization available");

    readLinear(is, value);

#if BOOST_ENDIAN_LITTLE_BYTE
    swapByteOrder(value);
#else
    // keep byte order
#endif
}

//==============================================================================
//! \brief Read a value of type bool from a stream.
//!
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<>
inline void readBE<bool>(std::istream& is, bool& value)
{
    uint8_t tmp;
    readLinear(is, tmp); //  Endianness does not matter for 1 Byte read.
    value = (tmp != 0);
}

//==============================================================================
//! \brief Read a 24 bit value from a stream into the uint32_t \a value.
//!        The 3 bytes are read in big-endian byte order from the input.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readBE24(std::istream& is, uint32_t& value)
{
    // Always read into target value as if the machine is big endian.
    // The first target byte (most significant byte in big endian) will
    // not be overridden and is still unchanged. It will set later to 0.
    char* targetBuf = reinterpret_cast<char*>(&value);
    is.read(targetBuf + 1, 3);

#if BOOST_ENDIAN_LITTLE_BYTE
    swapByteOrder(value);
#else
    // nothing to do
#endif

    // clear the most significant byte (independent of endianess)
    value = value & 0x00FFFFFFU;
}

//==============================================================================
//! \brief Read a 24 bit value from a stream into the int32_t \a value.
//!        The 3 bytes are read in big-endian byte order from the input.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
inline void readBE24(std::istream& is, int32_t& value)
{
    uint32_t& asUnsigned = *reinterpret_cast<uint32_t*>(&value);
    readBE24(is, asUnsigned);

    // most significant bit (here bit 23) set indicates negative number in 2-complement.
    if ((asUnsigned & 0x00800000U) == 0x00800000U)
    {
        // set the most significant byte in 2-complement for negative numbers.
        asUnsigned = asUnsigned | (0xFFU << 24);
    }
}

//==============================================================================
//! \brief Read a value of type \a T from a stream.
//!
//! \tparam T              Type of the value to be read.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//! \param[in]      sourceIsBigEndian
//!                        Set to \c true, if the stream has big-endian byte order
//!                        or to \c false, if it has little-endian byte order.
//------------------------------------------------------------------------------
template<typename T>
inline void read(std::istream& is, T& value, bool sourceIsBigEndian = false)
{
#if BOOST_ENDIAN_BIG_BYTE
    const bool do_swap = !sourceIsBigEndian;
#else
    const bool do_swap = sourceIsBigEndian;
#endif

    if (do_swap)
    {
        readSwapped(is, value);
    }
    else
    {
        readLinear(is, value);
    }
}

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Write a value of type \a T into a stream.
//!
//! It is assumed that the stream receiving the data in the same byte order
//! as the system the code is running on. So the byte order will be left
//! untouched.
//!
//! \tparam T  Type of the value to be written.
//! \param[in, out] os     Stream receiving the data been written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<typename T>
inline void writeLinear(std::ostream& os, const T& value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

//==============================================================================
//! \brief Write a value of type \a T into a stream.
//!
//! It is assumed that the stream receiving the data in different byte order
//! as the system the code is running on. So the byte order has to be swapped.
//!
//! \tparam T              Type of the value to be written.
//! \param[in, out] os     Stream receiving the data been written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeSwapped(std::ostream& os, const T& value)
{
    const T swappedValue = getWithSwappedByteOrder(value);

    os.write(reinterpret_cast<const char*>(&swappedValue), sizeof(T));
}

//==============================================================================
//! \brief Write a value of type \a T into a stream.
//!        Writing individual bytes is done in little-endian byte order.
//!
//! \tparam T              Type of the value to be written.
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<typename T>
inline void writeLE(std::ostream& os, const T& value)
{
    static_assert(std::is_fundamental<T>::value,
                  "Called writeLE template function with complex type. No specialization available");

#if BOOST_ENDIAN_BIG_BYTE
    writeSwapped(os, value);
#else
    writeLinear(os, value);
#endif
}

//==============================================================================
//! \brief Write a bool value into a stream.
//!
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<>
inline void writeLE<bool>(std::ostream& os, const bool& value)
{
    const uint8_t tmp = value ? 1 : 0;
    writeLinear(os, tmp); //  Endianness does not matter for 1 Byte read.
}

//==============================================================================
//! \brief Write a 24 bit value from a variable of type \a T into a stream.
//!        The 3 bytes are written in little-endian byte order the output.
//!
//! \tparam T              Type of the value to be written.
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeLE24(std::ostream& os, const T& value)
{
    static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
                  "Writing 24 bit values is implemented for types uint32_t or int32_t only!");

#if BOOST_ENDIAN_LITTLE_BYTE
    // Little endian start with least significant bytes.
    // Hence the last byte in value has to be ignored.
    // The first 3 bytes have to be written in linear order to keep LE.
    os.write(reinterpret_cast<const char*>(&value) + 0, 3);
#else
    // Big endian starts with the most significant byte.
    // Hence the first byte in value has to be ignored.
    // The last 3 bytes have to be written in reverse order to get LE.
    os.write(reinterpret_cast<const char*>(&value) + 3, 1);
    os.write(reinterpret_cast<const char*>(&value) + 2, 1);
    os.write(reinterpret_cast<const char*>(&value) + 1, 1);
#endif
}

//==============================================================================
//! \brief Write a value of type \a T into a stream.
//!        Writing individual bytes is done in big-endian byte order.
//!
//! \tparam T              Type of the value to be written.
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<typename T>
inline void writeBE(std::ostream& os, const T& value)
{
    static_assert(std::is_fundamental<T>::value,
                  "Called writeBE template function with complex type. No specialization available");
#if BOOST_ENDIAN_BIG_BYTE
    writeLinear(os, value);
#else
    writeSwapped(os, value);
#endif
}

//==============================================================================
//! \brief Write a bool value into a stream.
//!
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<>
inline void writeBE(std::ostream& os, const bool& value)
{
    const uint8_t tmp = value ? 1 : 0;
    writeLinear(os, tmp); //  Endianness does not matter for 1 Byte read.
}

//==============================================================================
//! \brief Write a 24 bit value from a variable of type \a T into a stream.
//!        The 3 bytes are written in big-endian byte order the output.
//!
//! \tparam T              Type of the value to be written.
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<typename T>
void writeBE24(std::ostream& os, const T& value)
{
    static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
                  "Writing 24 bit values is implemented for types uint32_t or int32_t only!");

#if BOOST_ENDIAN_LITTLE_BYTE
    // Little endian start with least significant bytes.
    // Hence the last byte in value has to be ignored.
    // The first 3 bytes have to be written in reverse order to get BE.
    os.write(reinterpret_cast<const char*>(&value) + 2, 1);
    os.write(reinterpret_cast<const char*>(&value) + 1, 1);
    os.write(reinterpret_cast<const char*>(&value) + 0, 1);
#else
    // big endian starts with the most significant byte.
    // Hence the first byte in value has to be ignored.
    // The last 3 bytes have to be written in linear order to keep BE.
    memcpy(targetBuf, reinterpret_cast<const char*>(&value) + 1, 3);
#endif
}

//==============================================================================
//! \brief Write a value of type \a T into a stream.
//!
//! \tparam T              Type of the value to be written.
//! \param[in, out] os     Stream the data will be written to.
//! \param[in]      value  The data to be written.
//! \param[in]      destIsBigEndian
//!                        Set to \c true, if the stream has big-endian byte order
//!                        or to \c false, if it has little-endian byte order.
//------------------------------------------------------------------------------
template<typename T>
inline void write(std::ostream& os, const T& value, bool destIsBigEndian = false)
{
#if BOOST_ENDIAN_BIG_BYTE
    const bool do_swap = !destIsBigEndian;
#else
    const bool do_swap = destIsBigEndian;
#endif

    if (do_swap)
    {
        writeSwapped(os, value);
    }
    else
    {
        writeLinear(os, value);
    }
}

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Bit number for reading/writing individual bits from/to a stream.
//------------------------------------------------------------------------------
enum class BitNumber : uint8_t
{
    Bit0 = 0,
    Bit1 = 1,
    Bit2 = 2,
    Bit3 = 3,
    Bit4 = 4,
    Bit5 = 5,
    Bit6 = 6,
    Bit7 = 7
};

//==============================================================================
//! \brief Write a value \a val with \a nbOfDataBits (between 1 and 8) bits to the given
//!        buffer \a buf starting at bit \a firstBitToWrite.
//!
//! If not all given bits fit into the byte pointed to by \a buf,
//! the rest of the bits will be written into the next byte. Starting
//! with the least significant bit.
//!
//! \param[in, out] buf              Pointer to the position of the byte where
//!                                  to write the first bit of \a val into.
//!                                  On exit this variable will point to the
//!                                  position inside of the buffer where to write
//!                                  from the next bit after this write operation.
//! \param[in, out] firstBitToWrite  Id of the first bit to be (over)written in
//!                                  the buffer.
//!                                  On exit this variable will hold the id of the
//!                                  next bit to be written to after this write
//!                                  operation.
//! \param[in]      val              Variable that contains the bit to be written
//!                                  into the buffer \a buf.
//! \param[in]      nbOfDataBits     The number of bits to be written into the buffer \a buf.
//!                                  These bits are provided in the parameter \a val. Starting
//!                                  with the least significant bit. \a nbOfDataBits must between 1 and 8.
//------------------------------------------------------------------------------
void writeLE(uint8_t*& buf, BitNumber& firstBitToWrite, uint8_t val, const int nbOfDataBits);

//==============================================================================

//==============================================================================
//! \brief Write a value \a val with \a nbOfDataBits (between 9 and 16) bits to the given
//!        buffer \a buf starting at bit \a firstBitToWrite.
//!
//! If not all given bits fit into the byte pointed to by \a buf,
//! the rest of the bits will be written into the next byte. Starting
//! with the least significant bit.
//!
//! \param[in, out] buf              Pointer to the position of the byte where
//!                                  to write the first bit of \a val into.
//!                                  On exit this variable will point to the
//!                                  position inside of the buffer where to write
//!                                  from the next bit after this write operation.
//! \param[in, out] firstBitToWrite  Id of the first bit to be (over)written in
//!                                  the buffer.
//!                                  On exit this variable will hold the id of the
//!                                  next bit to be written to after this write
//!                                  operation.
//! \param[in]      val              Variable that contains the bit to be written
//!                                  into the buffer \a buf.
//! \param[in]      nbOfDataBits     The number of bits to be written into the buffer \a buf.
//!                                  These bits are provided in the parameter \a val. Starting
//!                                  with the least significant bit. \a nbOfDataBits must between 9 and 16.
//------------------------------------------------------------------------------
void writeLE(uint8_t*& buf, BitNumber& firstBitToWrite, const uint16_t val, const int nbOfDataBits);

//==============================================================================
//! \brief Read a value \a val with \a nbOfDataBits (between 1 and 8) bits from the given
//!        buffer \a buf starting at bit \a firstBitToRead inside the buffer.
//!
//! If not all bits to be read available inside the byte pointed to by \a buf,
//! the rest of the bits will be read from the next byte. Starting
//! with the least significant bit.
//!
//! \param[in, out] buf              Pointer to the position of the byte where
//!                                  to read the first bit from.
//!                                  On exit this variable will point to the
//!                                  position inside of the buffer where to read
//!                                  the next bit from after this read operation.
//! \param[in, out] firstBitToRead   Id of the first bit to be read from
//!                                  the buffer.
//!                                  On exit this variable will hold the id of the
//!                                  next bit to be read from after this read
//!                                  operation.
//! \param[in]      nbOfDataBits     The number of bits to be read from the buffer \a buf.
//!                                  These bits are returned. \a nbOfDataBits must between 1 and 8.
//!
//! \return A byte that contains the read bits. Starting with the least significant bit.
//------------------------------------------------------------------------------
uint8_t readLE8(const uint8_t*& buf, BitNumber& firstBitToRead, const int nbOfDataBits);

//==============================================================================
//! \brief Read a value \a val with \a nbOfDataBits (between 9 and 16) bits from the given
//!        buffer \a buf starting at bit \a firstBitToRead inside the buffer.
//!
//! If not all bits to be read available inside the byte pointed to by \a buf,
//! the rest of the bits will be read from the next byte. Starting
//! with the least significant bit.
//!
//! \param[in, out] buf              Pointer to the position of the byte where
//!                                  to read the first bit from.
//!                                  On exit this variable will point to the
//!                                  position inside of the buffer where to read
//!                                  the next bit from after this read operation.
//! \param[in, out] firstBitToRead   Id of the first bit to be read from
//!                                  the buffer.
//!                                  On exit this variable will hold the id of the
//!                                  next bit to be read from after this read
//!                                  operation.
//! \param[in]      nbOfDataBits     The number of bits to be read from the buffer \a buf.
//!                                  These bits are returned. \a nbOfDataBits must between 9 and 16.
//!
//! \return A byte that contains the read bits. Starting with the least significant bit.
//------------------------------------------------------------------------------
uint16_t readLE16(const uint8_t*& buf, BitNumber& firstBitToRead, const int nbOfDataBits);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
