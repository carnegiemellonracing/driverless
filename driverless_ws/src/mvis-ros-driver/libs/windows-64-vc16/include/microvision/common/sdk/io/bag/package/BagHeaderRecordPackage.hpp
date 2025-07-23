//==============================================================================
//! \file
//!
//! \brief Data package to store BAG header record.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 06, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>
#include <microvision/common/sdk/io/bag/package/BagRecordPackage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package to store BAG header record.
//! \extends BagRecordPackage
//------------------------------------------------------------------------------
class BagHeaderRecordPackage : public BagRecordPackage
{
public:
    //========================================
    //! \brief BAG file header padding character.
    //----------------------------------------
    static constexpr char fileHeaderPaddingCharacter{0x20};

    //========================================
    //! \brief BAG file header padding space.
    //----------------------------------------
    static constexpr std::size_t fileHeaderPaddingSpace{4096};

    //========================================
    //! \brief BAG record header field name for index_pos.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameIndexPosition;

    //========================================
    //! \brief BAG record header field name for conn_count.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameConnectionCount;

    //========================================
    //! \brief BAG record header field name for chunk_count.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameChunkCount;

public:
    //========================================
    //! \brief Construct BAG header record with sequence index and source URI.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    BagHeaderRecordPackage(const int64_t index, const Uri& sourceUri);

    //========================================
    //! \brief Construct BAG header record with all posible parameters.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream) as Uri.
    //! \param[in] payload  BAG record data.
    //! \param[in] version  BAG format version
    //! \param[in] header   BAG record header.
    //----------------------------------------
    BagHeaderRecordPackage(const int64_t index,
                           const Uri& sourceUri,
                           const SharedBuffer& payload,
                           const BagFormatVersion version,
                           const BagRecordFieldMap& header);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagHeaderRecordPackage() override;

public: // getter
    //========================================
    //! \brief Offset of first record after the chunk section.
    //! \details The header key is 'index_pos' (index position).
    //! \returns Offset to first record after last chunk in stream.
    //----------------------------------------
    int64_t getPositionOfFirstRecordAfterChunkSection() const;

    //========================================
    //! \brief Number of unique connections in stream.
    //! \details The header key is 'conn_count' (connection count).
    //! \returns Number of connections.
    //----------------------------------------
    uint32_t getConnectionCount() const;

    //========================================
    //! \brief Number of chunk records in stream.
    //! \details The header key is 'chunk_count' (chunk count).
    //! \returns Number of chunks.
    //----------------------------------------
    uint32_t getChunkCount() const;

public: // setter
    //========================================
    //! \brief Offset of first record after the chunk section.
    //! \details The header key is 'index_pos' (index position).
    //! \param[in] value  Offset to first record after last chunk in stream.
    //----------------------------------------
    void setPositionOfFirstRecordAfterChunkSection(const int64_t value);

    //========================================
    //! \brief Number of unique connections in stream.
    //! \details The header key is 'conn_count' (connection count).
    //! \param[in] value  Count of connections.
    //----------------------------------------
    void setConnectionCount(const uint32_t value);

    //========================================
    //! \brief Number of chunk records in stream.
    //! \details The header key is 'chunk_count' (chunk count).
    //! \param[in] value  Number of chunks.
    //----------------------------------------
    void setChunkCount(const uint32_t value);

public:
    //========================================
    //! \brief BAG record has valid header fields.
    //! \return Either \c true if header has all guaranteed fields, otherwise \c false.
    //----------------------------------------
    bool isValid() const override;

    //========================================
    //! \brief Compute record size which includes header size.
    //! \return Static length of header record which is \c 4104.
    //----------------------------------------
    std::size_t computeRecordSize() const override;

}; // class BagHeaderRecordPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagHeaderRecordPackage pointer.
//------------------------------------------------------------------------------
using BagHeaderRecordPackagePtr = std::shared_ptr<BagHeaderRecordPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
