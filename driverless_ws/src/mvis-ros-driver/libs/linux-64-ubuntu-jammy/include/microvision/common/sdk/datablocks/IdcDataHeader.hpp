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
//! \date April 4, 2012
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Header for all idc messages/DataContainer sent via TCP/IP or stored in a
//!        IDC file.
//!
//! The IdcDataHeader is a uniform block of meta data that is prepended by the
//! MVIS SDK to all kinds of DataContainer at external communication as TCP/IP and
//! file output.
//!
//! The data of IdcDataHeader are related to the DataContainer, as DataType,
//! size and origin.
//!
//! The size of the previous message allows faster file navigation.
//!
//! And finally the timestamp says when the message (header) was generated/sent.
//------------------------------------------------------------------------------
class IdcDataHeader final
{
public: // constructor/destructor
    //========================================
    //! \brief Default constructor.
    //!
    //! Creates an invalid IdcDataHeader.
    //----------------------------------------
    IdcDataHeader();

    //========================================
    //! \brief Constructor.
    //!
    //! Creates data header and fills all required fields.
    //!
    //! \param[in] dataType           DataType of the DataContainer
    //!                               this header is for.
    //! \param[in] sizeOfPrevMessage  In a stream of DataBlocks,
    //!                               this is the size of the
    //!                               previous DataContainer
    //!                               (without the header).
    //! \param[in] sizeOfThisMessage  The size of the DataContainer
    //!                               this header is for.
    //! \param[in] deviceId           Id of the device which is
    //!                               the source of this message.
    //! \param[in] ntpTime            Timestamp of this header
    //!                               (not the DataContainer).
    //----------------------------------------
    IdcDataHeader(const DataTypeId dataType,
                  const uint32_t sizeOfPrevMessage,
                  const uint32_t sizeOfThisMessage,
                  const uint8_t deviceId,
                  const NtpTime ntpTime);

    //========================================
    //! \brief Destructor.
    //!
    //! Does nothing special.
    //----------------------------------------
    ~IdcDataHeader();

public:
    //========================================
    //! \brief Parse a stream and fill the attributes of this IdcDataHeader.
    //!
    //! Before reading the data from the stream the #magicWord
    //! will be searched. Only after reading the #magicWord, the
    //! data will be deserialized.
    //!
    //! \param[in, out] is  Stream to read the data from.
    //!                     On output the read pointer will
    //!                     point to the first byte behind
    //!                     the last byte belongs to the
    //!                     IdcDataHeader serialization.
    //! \return Whether the deserialization was successful.
    //! \retval true The deserialization was successful.
    //! \retval false Either no #magicWord has been found or
    //!               the method failed to read enough bytes
    //!               from the stream needed for the deserialization.
    //----------------------------------------
    bool deserialize(std::istream& is);

    //========================================
    //! \brief Write a serialization of this IdcDataHeader into the
    //!        output stream \a os.
    //!
    //! \param[in, out] os  Stream to write the serialization of
    //!                     this IdcDataHeader to.
    //! \return Whether the serialization was successful.
    //----------------------------------------
    bool serialize(std::ostream& os) const;

    //========================================
    //! \brief Get the serialized size of the IdcDataHeader.
    //!
    //! \return  Number of bytes used for the serialization.
    //----------------------------------------
    std::streamsize getSerializedSize() const { return getHeaderSize(); }

public:
    //========================================
    //! \brief Skip to next data header begin.
    //!
    //! Gobbles all byte from the stream
    //! before the next IdcDataHeader#magicWord.
    //!
    //! \param[in, out] is       The stream where to find the #magicWord in. If a #magicWord has been found,
    //!                          on exit the first 4 bytes in the stream will be the #magicWord.
    //! \return Whether a #magicWord has been found.
    //----------------------------------------
    static bool moveToMagicWord(std::istream& is);

    //========================================
    //! \brief Read backwards, trying to find a magic word.
    //!
    //! The stream will be stepped back until a magic word is found.
    //!
    //! \param[in, out ] is  Stream to look in for the magic word. On exit this
    //!                      stream will be empty if no magic word has been found
    //!                      or the read position will be before the magic word.
    //!
    //! \return Whether the magicWord has been read.
    //!
    //! \note This is slow and seeking backwards. And the magic word will have to be read again.
    //----------------------------------------
    static bool moveToPreviousMagicWord(std::istream& is);

public: // getter
    //========================================
    //! \brief Get the DataType of the associated
    //!        DataContainer.
    //! \return The DataType of the associated
    //!         DataContainer.
    //----------------------------------------
    DataTypeId getDataType() const { return m_dataType; }

    //========================================
    //! \brief Get the id of the source device of
    //!        the associated DataContainer.
    //! \return The id of the source device of
    //!         the associated DataContainer.
    //----------------------------------------
    uint8_t getDeviceId() const { return m_deviceId; }

    //========================================
    //! \brief Get the size of serialization of the
    //!        associated DataContainer.
    //! \return The size of serialization of the
    //!         associated DataContainer.
    //----------------------------------------
    uint32_t getMessageSize() const { return m_sizeOfThisMessage; }

    //========================================
    //! \brief Get the size of serialization of the
    //!        previous DataContainer.
    //! \return The size of serialization of the
    //!         previous DataContainer.
    //----------------------------------------
    uint32_t getPreviousMessageSize() const { return m_sizeOfPrevMessage; }

    //========================================
    //! \brief Get the timestamp of this IdcDataHeader.
    //! \return The timestamp of this IdcDataHeader.
    //----------------------------------------
    NtpTime getTimestamp() const { return m_ntpTime; }

public: // setter
    //========================================
    //! \brief Set the DataType of the associated
    //!        DataContainer.
    //! \param[in] dataType  The new dataType of the
    //!                      associated DataContainer.
    //----------------------------------------
    void setDataType(const DataTypeId dataType) { m_dataType = dataType; }

    //========================================
    //! \brief Set the id of the source device
    //!        of the associated DataContainer.
    //! \param[in] deviceId  The new id of the source device
    //!                      of the associated DataContainer.
    //----------------------------------------
    void setDeviceId(const uint8_t deviceId) { m_deviceId = deviceId; }

    //========================================
    //! \brief Set the size of the associated
    //!        DataContainer.
    //! \param[in] sizeOfThisMessage  The new size of the
    //!                               associated DataContainer.
    //----------------------------------------
    void setMessageSize(const uint32_t sizeOfThisMessage) { m_sizeOfThisMessage = sizeOfThisMessage; }

    //========================================
    //! \brief Set the size of the previous
    //!        DataContainer.
    //! \param[in] sizeOfPreviousMessage  The new  size of the
    //!                                   previous DataContainer.
    //----------------------------------------
    void setPreviousMessageSize(const uint32_t sizeOfPreviousMessage) { m_sizeOfPrevMessage = sizeOfPreviousMessage; }

    //========================================
    //! \brief Set the timestamp of the IdcDataHeader.
    //! \param[in] ntpTime  The new timestamp of the
    //!                     IdcDataHeader.
    //----------------------------------------
    void setTimestamp(const NtpTime ntpTime) { m_ntpTime = ntpTime; }

public:
    //========================================
    //! \brief Get the serialized size of an IdcDataHeader.
    //! \return The number of bytes used by the
    //!         serialization of IdcDataHeader.
    //----------------------------------------
    static constexpr uint8_t getHeaderSize() { return 24; }

private:
    //========================================
    //! \brief Gobbles all bytes from the input stream
    //!        till the magic word 0xAFFEC0C2 (big-endian)
    //!        has been found.
    //!
    //! The stream will be read until the magic word has been read. In case the magic word
    //! has been read \c true will be returned.
    //! The read position will be \b behind the magic word!
    //!
    //! \param[in, out ] is  Stream to look in for the magic word. On exit this
    //!                      stream will be empty if no magic word has been found
    //!                      or the read position will be \c behind the magic word.
    //!
    //! \return Whether the magicWord has been read.
    //!
    //! \note This is fast and reading forward only. Magic word is already read.
    //----------------------------------------
    static bool findNextMagicWord(std::istream& is);

public:
    static constexpr int64_t sizeOfMagicWord{4}; //!< In the header the first 4 bytes contain the magic word.

    //========================================
    //! \brief A sequence of 4 bytes that marks the start of a serialized IdcDataHeader.
    //!
    //! This sequence of bytes (0xAFFEC0C2) is used to mark the start of the header
    //! and be able to skip corrupt data until the next possible header.
    //----------------------------------------
    static constexpr const std::array<uint8_t, sizeOfMagicWord> magicWordBytes{0xAF, 0xFE, 0xC0, 0xC2};

    static constexpr uint32_t by1Byte{8U}; //!< used to shift numbers 8 bits (1 byte)
    static constexpr uint32_t by2Bytes{16U}; //!< used to shift numbers 16 bits (2 bytes)
    static constexpr uint32_t by3Bytes{24U}; //!< used to shift numbers 24 bits (3 bytes)

    //========================================
    //! \brief A sequence of 4 bytes that marks the start of a serialized IdcDataHeader.
    //!
    //! This sequence of bytes (0xAFFEC0C2) is used to mark the start of the header
    //! and be able to skip corrupt data until the next possible header.
    //!
    //! \note This is written in BE byte order!
    //----------------------------------------
    static constexpr uint32_t magicWord{
        (static_cast<uint32_t>(magicWordBytes[0]) << by3Bytes) // 0xAF
        + (static_cast<uint32_t>(magicWordBytes[1]) << by2Bytes) // 0xFE
        + (static_cast<uint32_t>(magicWordBytes[2]) << by1Byte) // 0xC0
        + static_cast<uint32_t>(magicWordBytes[3]) // 0xC2
    };

private:
    static constexpr const char* loggerId = "microvision::common::sdk::IdcDataHeader";
    static microvision::common::logging::LoggerSPtr logger;

private: // attributes
    //========================================
    //! \brief Number of bytes in the previous message.
    //----------------------------------------
    uint32_t m_sizeOfPrevMessage;

    //========================================
    //! \brief Number of bytes used by the
    //!        Serialization of the associated
    //!        DataContainer.
    //----------------------------------------
    uint32_t m_sizeOfThisMessage;

    //========================================
    //! \brief Id of the source of the associated
    //!        DataContainer
    //----------------------------------------
    uint8_t m_deviceId;

    //========================================
    //! \brief DataType of the associated DataContainer.
    //----------------------------------------
    DataTypeId m_dataType; // serialized as uint16_t

    //========================================
    //! \brief Timestamp of the IdcDataHeader.
    //----------------------------------------
    NtpTime m_ntpTime;
}; // IdcDataHeader

//==============================================================================

//==============================================================================
//! \brief Nullable IdcDataHeader pointer.
//------------------------------------------------------------------------------
using IdcDataHeaderPtr = std::shared_ptr<IdcDataHeader>;

//==============================================================================
//! \brief Equal predicate for IdcDataHeader.
//! \return \c True if \a lhs and \a rhs are equal.
//!         \c false otherwise.
//! \sa compareIdcDataHeaderWithoutDate(const IdcDataHeader& lhs, const IdcDataHeader& rhs)
//------------------------------------------------------------------------------
bool operator==(const IdcDataHeader& lhs, const IdcDataHeader& rhs);

//==============================================================================
//! \brief Inqual predicate for IdcDataHeader.
//! \return \c True if \a lhs and \a rhs are inequal.
//!         \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const IdcDataHeader& lhs, const IdcDataHeader& rhs);

//==============================================================================
//! \brief Equal predicate for IdcDataHeader omitting the timestamp entry.
//! \return \c True if \a lhs and \a rhs are equal (without timestamp)
//!         \c false otherwise.
//! \sa operator==(const IdcDataHeader& lhs, const IdcDataHeader& rhs)
//------------------------------------------------------------------------------
bool compareIdcDataHeaderWithoutDate(const IdcDataHeader& lhs, const IdcDataHeader& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
