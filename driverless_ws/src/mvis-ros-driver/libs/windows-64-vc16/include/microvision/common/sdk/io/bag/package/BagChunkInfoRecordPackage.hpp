//==============================================================================
//! \file
//!
//! \brief Data package to store BAG chunk info record.
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
#include <microvision/common/sdk/NtpTime.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package to store BAG chunk info record.
//! \extends BagRecordPackage
//------------------------------------------------------------------------------
class BagChunkInfoRecordPackage : public BagRecordPackage
{
public:
    //========================================
    //! \brief Info map for current version.
    //! \note Key is connection id
    //!       and value is count of data in chunk per connection.
    //----------------------------------------
    using InfoMap = std::map<uint32_t, uint32_t>;

public:
    //========================================
    //! \brief BAG record header field name for version.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameVersion;

    //========================================
    //! \brief BAG record header field name for chunk position.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameChunkPosition;

    //========================================
    //! \brief BAG record header field name for start time.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameStartTime;

    //========================================
    //! \brief BAG record header field name for end time.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameEndTime;

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
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagChunkInfoRecordPackage";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Construct BAG chunk info record with sequence index and source URI.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    BagChunkInfoRecordPackage(const int64_t index, const Uri& sourceUri);

    //========================================
    //! \brief Construct BAG chunk info record with all posible parameters.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream) as Uri.
    //! \param[in] payload  BAG record data.
    //! \param[in] version  BAG format version.
    //! \param[in] header   BAG record header.
    //----------------------------------------
    BagChunkInfoRecordPackage(const int64_t index,
                              const Uri& sourceUri,
                              const SharedBuffer& payload,
                              const BagFormatVersion version,
                              const BagRecordFieldMap& header);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagChunkInfoRecordPackage() override;

public: // getter
    //========================================
    //! \brief Chunk info record version.
    //! \details The header key is 'ver' (version).
    //! \returns Version of chunk info.
    //----------------------------------------
    uint32_t getVersion() const;

    //========================================
    //! \brief Offset of the chunk record in stream.
    //! \details The header key is 'chunk_pos' (chunk position).
    //! \returns Chunk record offset.
    //----------------------------------------
    int64_t getChunkPosition() const;

    //========================================
    //! \brief Timestamp of earliest message in the chunk.
    //! \details The header key is 'start_time' (start time).
    //! \returns First timestamp in chunk.
    //----------------------------------------
    Ptp64Time getFirstTimestamp() const;

    //========================================
    //! \brief Timestamp of latest message in the chunk.
    //! \details The header key is 'end_time' (end time).
    //! \returns Last timestamp in chunk.
    //----------------------------------------
    Ptp64Time getLastTimestamp() const;

    //========================================
    //! \brief Number of connections in the chunk.
    //! \details The header key is 'count' (count).
    //! \returns Number of connections in chunk.
    //----------------------------------------
    uint32_t getConnectionCount() const;

public: // setter
    //========================================
    //! \brief Chunk info record version.
    //! \details The header key is 'ver' (version).
    //! \param[in] value  Version of chunk info.
    //----------------------------------------
    void setVersion(const uint32_t value);

    //========================================
    //! \brief Offset of the chunk record in stream.
    //! \details The header key is 'chunk_pos' (chunk position).
    //! \param[in] value  Chunk record offset.
    //----------------------------------------
    void setChunkPosition(const int64_t value);

    //========================================
    //! \brief Timestamp of earliest message in the chunk.
    //! \details The header key is 'start_time' (start time).
    //! \param[in] value  First timestamp in chunk.
    //----------------------------------------
    void setFirstTimestamp(const Ptp64Time& value);

    //========================================
    //! \brief Timestamp of latest message in the chunk.
    //! \details The header key is 'end_time' (end time).
    //! \param[in] value  Last timestamp in chunk.
    //----------------------------------------
    void setLastTimestamp(const Ptp64Time& value);

    //========================================
    //! \brief Number of connections in the chunk.
    //! \details The header key is 'count' (count).
    //! \param[in] value  Number of connections in chunk.
    //----------------------------------------
    void setConnectionCount(const uint32_t value);

public:
    //========================================
    //! \brief BAG record has valid header fields.
    //! \return Either \c true if header has all guaranteed fields, otherwise \c false.
    //----------------------------------------
    bool isValid() const override;

public:
    //========================================
    //! \brief Read info of current version.
    //! \param[out] infos  Info map
    //! \note Key of info map is connection id
    //!       and value is count of data in chunk per connection.
    //! \return Either \c true if info is of current version
    //!         and successful readed, otherwise \c false.
    //----------------------------------------
    bool readInfos(InfoMap& infos);

    //========================================
    //! \brief Read info of current version.
    //! \param[in] infos  Info map
    //! \note Key of info map is connection id
    //!       and value is count of data in chunk per connection.
    //! \return Either \c true if info is of current version
    //!         and successful written, otherwise \c false.
    //----------------------------------------
    bool writeInfos(const InfoMap& infos);

}; // class BagChunkInfoRecordPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagChunkInfoRecordPackage pointer.
//------------------------------------------------------------------------------
using BagChunkInfoRecordPackagePtr = std::shared_ptr<BagChunkInfoRecordPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
