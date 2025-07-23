//==============================================================================
//! \file
//!
//! \brief Write BagDataPackages to BAG stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 11, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>
#include <microvision/common/sdk/io/bag/BagStreamWriterBase.hpp>
#include <microvision/common/sdk/io/bag/package/BagDataPackage.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements functionality to write BagDataPackages to BAG stream.
//! \note The packages will be written in batches
//!       triggered by time duration, size or manually.
//------------------------------------------------------------------------------
struct BagDataHeaderHashGlobal
{
    //========================================
    //! \brief Compute hash to uniquely identify data header.
    //! \note Combines topic, data type and message definition checksum to a unqiue hash.
    //! \returns Hash of data header.
    //----------------------------------------
    std::size_t operator()(const BagDataHeaderPtr& header) const
    {
        std::size_t hash{0};
        if (header)
        {
            boost::hash_combine(hash, stringHasher(header->getTopic()));
            boost::hash_combine(hash, stringHasher(header->getDataType()));
            boost::hash_combine(hash, stringHasher(header->getMessageDefinitionMd5Checksum()));
        }
        return hash;
    }

    //========================================
    //! \brief String hash generator.
    //----------------------------------------
    std::hash<std::string> stringHasher{};

}; // struct BagDataHeaderHashGlobal

class BagDataStreamWriter : public BagStreamWriterBase
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using BagStreamWriterBase::open;

private:
    //========================================
    //! \brief Logger name for logging configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagDataStreamWriter";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

private:
    //========================================
    //! \brief Meta information of chunk.
    //----------------------------------------
    struct ChunkInfo
    {
        //========================================
        //! \brief Index at which point in stream the chunk begins.
        //----------------------------------------
        int64_t index;

        //========================================
        //! \brief Earliest time in chunk.
        //----------------------------------------
        Ptp64Time firstTime;

        //========================================
        //! \brief Latest time in chunk.
        //----------------------------------------
        Ptp64Time lastTime;

        //========================================
        //! \brief Pointer to record data package of chunk info.
        //----------------------------------------
        BagChunkInfoRecordPackagePtr data;

        //========================================
        //! \brief Map of connection id
        //!        and count of data in chunk related to this connection.
        //----------------------------------------
        BagChunkInfoRecordPackage::InfoMap connectionInfos;

    }; // struct ChunkInfo

    //========================================
    //! \brief Compute hash to uniquely identify data header.
    //!
    //! Combines topic, data type and message definition checksum to a unqiue hash.
    //----------------------------------------
    struct BagDataHeaderHash
    {
        //========================================
        //! \brief Compute hash to uniquely identify data header.
        //! \note Combines topic, data type and message definition checksum to a unqiue hash.
        //! \returns Hash of data header.
        //----------------------------------------
        std::size_t operator()(const BagDataHeaderPtr& header) const;

        //========================================
        //! \brief String hash generator.
        //----------------------------------------
        std::hash<std::string> stringHasher{};

    }; // struct BagDataHeaderHash

    //========================================
    //! \brief Hash map of data header to connection record data package.
    //----------------------------------------
    using ConnectionMap = std::unordered_map<BagDataHeaderPtr, BagConnectionRecordPackagePtr, BagDataHeaderHashGlobal>;

    //========================================
    //! \brief Hash map of data header to chunk data time indices.
    //----------------------------------------
    using IndexMap = std::unordered_map<BagDataHeaderPtr, BagIndexRecordPackage::IndexMap, BagDataHeaderHashGlobal>;

public:
    //========================================
    //! \brief Default constructor which requires an uri of format BAG.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit BagDataStreamWriter(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagDataStreamWriter(BagDataStreamWriter&& writer);

    //========================================
    //! \brief Copy constructor is disabled because of stream dependency.
    //----------------------------------------
    BagDataStreamWriter(const BagDataStreamWriter&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagDataStreamWriter(void) override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    BagDataStreamWriter& operator=(BagDataStreamWriter&& writer);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagDataStreamWriter& operator=(const BagDataStreamWriter& writer) = delete;

public:
    //========================================
    //! \brief Set size trigger when to write the current batch of data to stream.
    //! \param[in] chunkSize  Maximal size of data in chunk.
    //----------------------------------------
    void flushChunkIfSizeIsReached(const std::size_t chunkSize);

    //========================================
    //! \brief Set duration trigger when to write the current batch of data to stream.
    //! \param[in] chunkDuration  Maximal duration between earliest and lastest data in chunk.
    //----------------------------------------
    void flushChunkIfDurationIsReached(const Ptp64Time& chunkDuration);

    //========================================
    //! \brief Write the current batch of data to stream.
    //! \returns Either \c true if data is written successful in stream, otherwise \c false.
    //----------------------------------------
    bool flushChunk();

public: // implements DataPackageStreamWriter
    //========================================
    //! \brief Request resource access to append data if possible.
    //! \param[in] ioStream  Resource stream to move.
    //! \param[in] append    Either \c true if the resource requested is in append mode, otherwise \c false.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream, const bool appendMode) override;

    //========================================
    //! \brief Release resources and stream ownership.
    //! \returns Get stream ownership back.
    //----------------------------------------
    IoStreamPtr release() override;

public: // implements DataPackageStreamWriter
    //========================================
    //! \brief Write DataPackage at stream position.
    //! \param[in/out] data  DataPackage to write.
    //! \returns \c true if DataPackage successful written, otherwise \c false.
    //----------------------------------------
    bool writePackage(DataPackage& dataPackage) override;

    //========================================
    //! \brief Write data package at stream position.
    //! \param[in] data  DataPackage to write.
    //! \returns Either \c true if DataPackage successful written, otherwise \c false.
    //!
    //! \note This method does not change source and index in the package header.
    //!       That data is required for serialization but possibly not for your code.
    //----------------------------------------
    bool writePackage(const DataPackage& dataPackage) override;

private:
    //========================================
    //! \brief Create record packages for data package and cache them in current batch.
    //! \param[in] bagPackage  BAG data package which is to write.
    //! \returns Either \c true if data is cached successful, otherwise \c false.
    //----------------------------------------
    bool storeBagDataPackage(BagDataPackage& bagPackage);

    //========================================
    //! \brief Index existing chunk infos and connections in preparation for appending to stream.
    //! \note chunk infos and connection will be overwritten by next bunch of data.
    //! \returns Either \c true if successful index, otherwise \c false.
    //----------------------------------------
    bool prepareAppend();

private:
    //========================================
    //! \brief Meta infomation of all chunks in stream.
    //----------------------------------------
    std::vector<ChunkInfo> m_chunkInfos;

    //========================================
    //! \brief All connections in stream.
    //----------------------------------------
    ConnectionMap m_connections;

    //========================================
    //! \brief Greatest indexed connection id to solve uniqueness conflicts.
    //!
    //! Will count up if a connection id is already in use.
    //----------------------------------------
    uint32_t m_lastConnectionId;

    //========================================
    //! \brief Current meta information of cached chunk.
    //----------------------------------------
    ChunkInfo m_currentChunkInfo;

    //========================================
    //! \brief Current size of cached chunk.
    //----------------------------------------
    std::size_t m_currentChunkSize;

    //========================================
    //! \brief Current indecies of cached chunk.
    //----------------------------------------
    IndexMap m_currentChunkIndexes;

    //========================================
    //! \brief Current connections used in cached chunk.
    //----------------------------------------
    ConnectionMap m_currentChunkConnections;

    //========================================
    //! \brief Current data in cached chunk.
    //----------------------------------------
    std::vector<BagRecordPackagePtr> m_currentChunkData;

    //========================================
    //! \brief Duration trigger, when to write current chunk.
    //----------------------------------------
    Ptp64Time m_flushChunkIfDurationIsReached;

    //========================================
    //! \brief Size trigger, when to write current chunk.
    //----------------------------------------
    std::size_t m_flushChunkIfSizeIsReached;

}; // class BagDataStreamWriter

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
