//==============================================================================
//! \file
//!
//! \brief Data package to store BAG index record.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 07, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>

#include <microvision/common/sdk/io/bag/package/BagRecordPackage.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package to store BAG index record.
//! \extends BagRecordPackage
//------------------------------------------------------------------------------
class BagIndexRecordPackage : public BagRecordPackage
{
public:
    //========================================
    //! \brief Index map for current version.
    //! \note The key is time as numeric
    //!       and the value is the position in decompressed chunk.
    //----------------------------------------
    using IndexMap = std::multimap<Ptp64Time, uint32_t>;

public:
    //========================================
    //! \brief BAG record header field name for version.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameVersion;

    //========================================
    //! \brief BAG record header field name for connection.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameConnection;

    //========================================
    //! \brief BAG record header field name for count.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameCount;

    //========================================
    //! \brief Current index version.
    //----------------------------------------
    static constexpr uint8_t currentIndexVersion{0x01U};

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagIndexRecordPackage";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Construct BAG index record with sequence index and source URI.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    BagIndexRecordPackage(const int64_t index, const Uri& sourceUri);

    //========================================
    //! \brief Construct BAG index record with all posible parameters.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream) as Uri.
    //! \param[in] payload  BAG record data.
    //! \param[in] version  BAG format version
    //! \param[in] header   BAG record header.
    //----------------------------------------
    BagIndexRecordPackage(const int64_t index,
                          const Uri& sourceUri,
                          const SharedBuffer& payload,
                          const BagFormatVersion version,
                          const BagRecordFieldMap& header);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagIndexRecordPackage() override;

public: // getter
    //========================================
    //! \brief Index data record version.
    //! \details The header key is 'ver' (version).
    //! \returns Version of index data.
    //----------------------------------------
    uint32_t getVersion() const;

    //========================================
    //! \brief ID of connection on which message arrived.
    //! \details The header key is 'conn' (connection).
    //! \returns The id of the connection.
    //----------------------------------------
    uint32_t getConnectionId() const;

    //========================================
    //! \brief Number of messages on connection in the preceding chunk.
    //! \details The header key is 'count' (count).
    //! \returns Number of messages in chunk.
    //----------------------------------------
    uint32_t getCount() const;

public: // setter
    //========================================
    //! \brief Index data record version.
    //! \details The header key is 'ver' (version).
    //! \param[in] value  Version of index data.
    //----------------------------------------
    void setVersion(const uint32_t value);

    //========================================
    //! \brief ID of connection on which message arrived.
    //! \details The header key is 'conn' (connection).
    //! \param[in] value  The id of the connection.
    //----------------------------------------
    void setConnectionId(const uint32_t value);

    //========================================
    //! \brief Number of messages on connection in the preceding chunk.
    //! \details The header key is 'count' (count).
    //! \param[in] value  Number of messages in chunk.
    //----------------------------------------
    void setCount(const uint32_t value);

public:
    //========================================
    //! \brief BAG record has valid header fields.
    //! \return Either \c true if header has all guaranteed fields, otherwise \c false.
    //----------------------------------------
    bool isValid() const override;

public:
    //========================================
    //! \brief Read index of current version.
    //! \param[out] index  Index map
    //! \note The key of index map is time as numeric
    //!       and the value is the position in decompressed chunk.
    //! \return Either \c true if index is of current version
    //!         and successful readed, otherwise \c false.
    //----------------------------------------
    bool readIndices(IndexMap& index);

    //========================================
    //! \brief Read index of current version.
    //! \param[in] index  Index map
    //! \note The key of index map is time as numeric
    //!       and the value is the position in decompressed chunk.
    //! \return Either \c true if index is of current version
    //!         and successful written, otherwise \c false.
    //----------------------------------------
    bool writeIndices(const IndexMap& index);

}; // class BagIndexRecordPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagIndexRecordPackage pointer.
//------------------------------------------------------------------------------
using BagIndexRecordPackagePtr = std::shared_ptr<BagIndexRecordPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
