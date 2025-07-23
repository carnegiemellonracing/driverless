//==============================================================================
//! \file
//!
//! \brief Data package to store BAG record.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 04, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>

#include <microvision/common/sdk/io/bag/package/BagRecordFieldMap.hpp>
#include <microvision/common/sdk/io/DataPackage.hpp>
#include <microvision/common/sdk/io.hpp>

#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Enum of BAG file format versions.
//------------------------------------------------------------------------------
enum class BagFormatVersion : uint8_t
{
    Invalid, //!< Invalid version
    Version20, //!< Version 2.0
};

//==============================================================================
//! \brief Read a value of type \a BagFormatVersion from a stream.
//!        Reading individual bytes is done in little endian byte order.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<>
void readLE<BagFormatVersion>(std::istream& is, BagFormatVersion& value);

//==============================================================================
//! \brief Read a value of type \a BagFormatVersion from a stream.
//!        Reading individual bytes is done in little endian byte order.
//! \param[in, out] is     Stream providing the data to be read.
//! \return The value that has been read.
//------------------------------------------------------------------------------
template<>
BagFormatVersion readLE<BagFormatVersion>(std::istream& is);

//==============================================================================
//! \brief Write a value of type \a BagFormatVersion into a stream.
//!        Writing individual bytes is done in little endian byte order.
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<>
void writeLE<BagFormatVersion>(std::ostream& os, const BagFormatVersion& value);

//==============================================================================
//! \brief Write human readable repräsentation of version in stream.
//! \param[in, out] stream      Output stream
//! \param[in]      version     BAG format version
//! \return The outputstream for operator chain.
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& stream, const BagFormatVersion version);

//==============================================================================
//! \brief Enum of BAG record types.
//------------------------------------------------------------------------------
enum class BagRecordType : uint8_t
{
    Unknown    = 0x00U, //!< Unkown record type
    Header     = 0x03U, //!< Bag file header, has to be at start of stream
    Chunk      = 0x05U, //!< Chunk of data messages and connections
    Connection = 0x07U, //!< Connection info aka topic
    Message    = 0x02U, //!< Data message with the serialized data
    Index      = 0x04U, //!< Time indexed data messages in chunks per connection
    ChunkInfo  = 0x06U //!< Header informations about a chunk
};

//==============================================================================
//! \brief Read a value of type \a BagRecordType from a stream.
//!        Reading individual bytes is done in little endian byte order..
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<>
void readLE<BagRecordType>(std::istream& is, BagRecordType& value);

//==============================================================================
//! \brief Read a value of type \a BagRecordType from a stream.
//!        Reading individual bytes is done in little endian byte order.
//! \param[in, out] is     Stream providing the data to be read.
//! \return The value that has been read.
//------------------------------------------------------------------------------
template<>
BagRecordType readLE<BagRecordType>(std::istream& is);

//==============================================================================
//! \brief Write a value of type \a BagRecordType into a stream.
//!        Writing individual bytes is done in little endian byte order.
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<>
void writeLE<BagRecordType>(std::ostream& os, const BagRecordType& value);

//==============================================================================
//! \brief Write human readable representation of record type in stream.
//! \param[in, out] stream      Output stream
//! \param[in]      type        BAG Record type
//! \return The outputstream for operator chain.
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& stream, const BagRecordType type);

//==============================================================================
//! \brief Data package to store BAG record.
//! \extends DataPackage
//------------------------------------------------------------------------------
class BagRecordPackage : public DataPackage
{
public:
    //========================================
    //! \brief BAG file record header field name for operation.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameOperation;

public:
    //========================================
    //! \brief Construct BAG record with sequence index and source URI.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    BagRecordPackage(const int64_t index, const Uri& sourceUri);

    //========================================
    //! \brief Construct BAG record with all possible parameters.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream) as Uri.
    //! \param[in] payload  BAG record data.
    //! \param[in] version  BAG format version
    //! \param[in] header   BAG record header.
    //----------------------------------------
    BagRecordPackage(const int64_t index,
                     const Uri& sourceUri,
                     const SharedBuffer& payload,
                     const BagFormatVersion version,
                     const BagRecordFieldMap& header);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagRecordPackage() override;

public:
    //========================================
    //! \brief Compare two bag record packages for equality.
    //! \param[in] lhs  BagRecordPackage to compare.
    //! \param[in] rhs  BagRecordPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const BagRecordPackage& lhs, const BagRecordPackage& rhs);

    //========================================
    //! \brief Compare two bag record packages for inequality.
    //! \param[in] lhs  BagRecordPackage to compare.
    //! \param[in] rhs  BagRecordPackage to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const BagRecordPackage& lhs, const BagRecordPackage& rhs);

public: // getter
    //========================================
    //! \brief Get BAG format version.
    //! \returns Version of BAG format.
    //----------------------------------------
    BagFormatVersion getFormatVersion() const;

    //========================================
    //! \brief Get key value map of record header.
    //! \returns Key value map.
    //----------------------------------------
    BagRecordFieldMap& getHeader();

    //========================================
    //! \brief Get key value map of record header.
    //! \returns Key value map.
    //----------------------------------------
    const BagRecordFieldMap& getHeader() const;

    //========================================
    //! \brief Get BAG record type from header.
    //! \details The header key is 'op' (operation).
    //! \returns BAG record type.
    //----------------------------------------
    BagRecordType getType() const;

public: // setter
    //========================================
    //! \brief Set BAG format version.
    //! \param[in] version  Of BAG format.
    //----------------------------------------
    void setFormatVersion(const BagFormatVersion version);

    //========================================
    //! \brief Set key value map of record header.
    //! \param[in] header  Key value map of header.
    //----------------------------------------
    void setHeader(const BagRecordFieldMap& header);

    //========================================
    //! \brief Set BAG record type on header.
    //! \details The header key is 'op' (operation).
    //! \param[in] type  BAG record type.
    //----------------------------------------
    void setType(const BagRecordType type);

public:
    //========================================
    //! \brief BAG record has valid header fields.
    //! \return Either \c true if header has all guaranteed fields, otherwise \c false.
    //----------------------------------------
    virtual bool isValid() const;

    //========================================
    //! \brief Compute record size which includes header size.
    //! \return Length of record or \c 8 if record is empty.
    //----------------------------------------
    virtual std::size_t computeRecordSize() const;

private:
    //========================================
    //! \brief BAG format version.
    //----------------------------------------
    BagFormatVersion m_version;

    //========================================
    //! \brief BAG record header.
    //----------------------------------------
    BagRecordFieldMap m_header;

}; // class BagRecordPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagRecordPackage pointer.
//------------------------------------------------------------------------------
using BagRecordPackagePtr = std::shared_ptr<BagRecordPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
