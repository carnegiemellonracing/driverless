//==============================================================================
//! \file
//!
//! \brief Data package to store BAG chunk record.
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
//! \brief Data package to store BAG chunk record.
//! \extends BagRecordPackage
//------------------------------------------------------------------------------
class BagChunkRecordPackage : public BagRecordPackage
{
public:
    //========================================
    //! \brief BAG record header field name for compression.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameCompression;

    //========================================
    //! \brief BAG record header field name for size.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameSize;

    //========================================
    //! \brief BAG chunk record default compression type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string defaultChunkRecordCompressionType;

    //========================================
    //! \brief BAG chunk record bzip2 compression type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string bzip2ChunkRecordCompressionType;

public:
    //========================================
    //! \brief Construct BAG chunk record with sequence index and source URI.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    BagChunkRecordPackage(const int64_t index, const Uri& sourceUri);

    //========================================
    //! \brief Construct BAG chunk record with all posible parameters.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream) as Uri.
    //! \param[in] payload  BAG record data.
    //! \param[in] version  BAG format version.
    //! \param[in] header   BAG record header.
    //----------------------------------------
    BagChunkRecordPackage(const int64_t index,
                          const Uri& sourceUri,
                          const SharedBuffer& payload,
                          const BagFormatVersion version,
                          const BagRecordFieldMap& header);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagChunkRecordPackage() override;

public: // getter
    //========================================
    //! \brief Compression type for the data.
    //! \details The header key is 'compression' (compression).
    //! \returns Compression type.
    //----------------------------------------
    std::string getCompressionType() const;

    //========================================
    //! \brief Size in bytes of the uncompressed chunk.
    //! \details The header key is 'size' (uncompressed chunk size).
    //! \returns Uncompressed chunk size.
    //----------------------------------------
    uint32_t getSizeOfUncompressedPayload() const;

public: // setter
    //========================================
    //! \brief Compression type for the data.
    //! \details The header key is 'compression' (compression).
    //! \param[in] value  Compression type.
    //----------------------------------------
    void setCompressionType(const std::string& value);

    //========================================
    //! \brief Size in bytes of the uncompressed chunk.
    //! \details The header key is 'size' (uncompressed chunk size).
    //! \param[in] value  Uncompressed chunk size.
    //----------------------------------------
    void setSizeOfUncompressedPayload(const uint32_t value);

public:
    //========================================
    //! \brief BAG record has valid header fields.
    //! \return Either \c true if header has all guaranteed fields, otherwise \c false.
    //----------------------------------------
    bool isValid() const override;

    //========================================
    //! \brief Compute record size which includes header size.
    //! \return Length of uncompressed chunk record.
    //----------------------------------------
    std::size_t computeRecordSize() const override;

}; // class BagChunkRecordPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagChunkRecordPackage pointer.
//------------------------------------------------------------------------------
using BagChunkRecordPackagePtr = std::shared_ptr<BagChunkRecordPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
