//==============================================================================
//! \file
//!
//! \brief Read BagDataPackages from BAG stream in chronological order.
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
#include <microvision/common/sdk/io/bag/BagStreamReaderBase.hpp>
#include <microvision/common/sdk/io/bag/package/BagDataPackage.hpp>
#include <microvision/common/sdk/io/bag/BagRecordChunkStreamReader.hpp>

#include <map>
#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Implements functionality to read BagDataPackages from BAG stream.
//!
//! Provides functions to read forward/backward in BAG stream.
//!
//! \note The packages will be read in chronological order.
//------------------------------------------------------------------------------
class BagDataStreamReader final : public BagStreamReaderBase
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using BagStreamReaderBase::open;

    //========================================
    //! \brief Predicate function to filter data by meta infomation.
    //----------------------------------------
    using ConnectionFilter = std::function<bool(const BagDataHeader&)>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagDataStreamReader";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Time sorted map of data index position in chunk.
    //----------------------------------------
    using TimeIndexMap = std::multimap<Ptp64Time, int64_t>;

    //========================================
    //! \brief Nullable TimeIndexMap pointer.
    //----------------------------------------
    using TimeIndexMapPtr = std::shared_ptr<TimeIndexMap>;

    //========================================
    //! \brief Nullable BagRecordChunkStreamReader pointer.
    //----------------------------------------
    using ChunkStreamReaderPtr = std::shared_ptr<BagRecordChunkStreamReader>;

    //========================================
    //! \brief Meta information of BAG file.
    //----------------------------------------
    struct HeaderInfo
    {
        //========================================
        //! \brief Index at which point in stream
        //!        the connections and chunk infos appear first.
        //----------------------------------------
        int64_t indexSection;

        //========================================
        //! \brief Lowest time in stream.
        //----------------------------------------
        Ptp64Time firstTimeInStream;

        //========================================
        //! \brief Highest time in stream.
        //----------------------------------------
        Ptp64Time lastTimeInStream;

    }; // struct HeaderInfo

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
        //! \brief Count of connection infos.
        //----------------------------------------
        uint32_t connectionCount;

        //========================================
        //! \brief Map of connection id
        //!        and count of data in chunk related to this connection.
        //----------------------------------------
        BagChunkInfoRecordPackage::InfoMap connectionInfos;

    }; // struct ChunkInfo

    //========================================
    //! \brief Compare two chunk infos for equality.
    //! \param[in] lhs  ChunkInfo to compare.
    //! \param[in] rhs  ChunkInfo to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const ChunkInfo& lhs, const ChunkInfo& rhs);

    //========================================
    //! \brief Compare two chunk infos for inequality.
    //! \param[in] lhs  ChunkInfo to compare.
    //! \param[in] rhs  ChunkInfo to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const ChunkInfo& lhs, const ChunkInfo& rhs);

    //========================================
    //! \brief Get hash to unique identify chunk info.
    //! \note Returns the chunk index, which should be unqiue in stream.
    //----------------------------------------
    struct ChunkInfoHash
    {
        //========================================
        //! \brief Get the unique identifier of chunk info used as hash.
        //! \note Returns the chunk index, which should be unique in stream.
        //! \param[in] value  Chunk info to hash.
        //! \returns Chunk index as chunk info hash.
        //----------------------------------------
        std::size_t operator()(const ChunkInfo& value) const;

    }; // struct ChunkInfoHash

    //========================================
    //! \brief Read context of chunk.
    //----------------------------------------
    struct ChunkData
    {
        //========================================
        //! \brief Pointer to chunk stream reader.
        //----------------------------------------
        ChunkStreamReaderPtr reader;

        //========================================
        //! \brief Pointer to chunk record package.
        //----------------------------------------
        BagChunkRecordPackagePtr data;

        //========================================
        //! \brief Pointer to time sorted map of chunk indices.
        //----------------------------------------
        TimeIndexMapPtr recordTimeIndex;

        //========================================
        //! \brief Iterator pointer to next data index.
        //----------------------------------------
        TimeIndexMap::const_iterator currentRecordIndex;

    }; // struct ChunkData

    //========================================
    //! \brief Key used by chunk info time map.
    //----------------------------------------
    struct ChunkTimeBoundary
    {
        //========================================
        //! \brief Chunk info first or last time.
        //----------------------------------------
        Ptp64Time time;

        //========================================
        //! \brief Either \c true if first time of chunk,
        //!        otherwise \c false if last time of chunk.
        //----------------------------------------
        bool lowerBounding;

    }; // struct ChunkTimeBoundary

    //========================================
    //! \brief Comparator whether chunk info time boundary is less (<) than other.
    //! \note If time of boundary is equal the lower boundary is lesser.
    //----------------------------------------
    struct ChunkTimeBoundaryLessCompare
    {
        //========================================
        //! \brief Comparator of chunk info time bounding is lesser (<) than another bounding.
        //! \note If time is bounding is equals the lower bounding is lesser.
        //! \param[in] lhs  Left ChunkTimeBoundary of operator.
        //! \param[in] rhs  Right ChunkTimeBoundary of operator.
        //! \returns Either \c true if lhs time of bounding is lesser (<) than rhs, otherwise \c false.
        //----------------------------------------
        bool operator()(const ChunkTimeBoundary& lhs, const ChunkTimeBoundary& rhs) const;

    }; // struct ChunkTimeBoundaryLessCompare

    //========================================
    //! \brief Sorted map of chunk info time boundings to chunk infos.
    //----------------------------------------
    using ChunkInfoTimeScale = std::multimap<ChunkTimeBoundary, ChunkInfo, ChunkTimeBoundaryLessCompare>;

    //========================================
    //! \brief Unsorted map of chunk infos to chunk read context.
    //----------------------------------------
    using ChunkDataMap = std::unordered_map<ChunkInfo, ChunkData, ChunkInfoHash>;

    //========================================
    //! \brief Map of sorted connection ids to data meta information.
    //----------------------------------------
    using ConnectionDataMap = std::map<uint32_t, BagDataHeaderPtr>;

    //========================================
    //! \brief Bidirectional iterator on ChunkInfoTimeScale.
    //!
    //!  This differs to normal iterator operator:
    //!     - end() - (Changed) Returns an iterator to the last element instead of eof.
    //!     - eof() - (Added) Returns out of bounds iterator like end() normally.
    //!     - --it  - (Changed) The backward operator of begin returns eof instead of begin.
    //----------------------------------------
    class ChunkInfoTimeScaleIterator final
    {
    public:
        //========================================
        //! \brief Move constructor.
        //! \param[in, out] other  Other iterator to move.
        //----------------------------------------
        ChunkInfoTimeScaleIterator(ChunkInfoTimeScaleIterator&& other);

        //========================================
        //! \brief Copy constructor.
        //! \param[in, out] other  Other iterator to copy.
        //----------------------------------------
        ChunkInfoTimeScaleIterator(const ChunkInfoTimeScaleIterator& other);

        //========================================
        //! \brief Construct iterator by ChunkInfoTimeScale.
        //! \note The position of iterator will set on eof().
        //! \param[in, out] timeScale  Map to iterate on.
        //----------------------------------------
        explicit ChunkInfoTimeScaleIterator(ChunkInfoTimeScale& timeScale);

        //========================================
        //! \brief Construct iterator by ChunkInfoTimeScale and start position.
        //! \param[in, out] timeScale           Map to iterate on.
        //! \param[in]      currentPosition     Position at which to start.
        //----------------------------------------
        ChunkInfoTimeScaleIterator(ChunkInfoTimeScale& timeScale, ChunkInfoTimeScale::iterator currentPosition);

        //========================================
        //! \brief Move assign operator.
        //! \param[in, out] other  Other iterator to move.
        //! \returns Reference on this.
        //----------------------------------------
        ChunkInfoTimeScaleIterator& operator=(ChunkInfoTimeScaleIterator&& other);

        //========================================
        //! \brief Copy assign operator.
        //! \param[in, out] other  Other iterator to copy.
        //! \returns Reference on this.
        //----------------------------------------
        ChunkInfoTimeScaleIterator& operator=(const ChunkInfoTimeScaleIterator& other);

        //========================================
        //! \brief Compare two iterators for equality.
        //! \param[in] lhs  ChunkInfoTimeScaleIterator to compare.
        //! \param[in] rhs  ChunkInfoTimeScaleIterator to compare.
        //! \returns Either \c true if equals or otherwise \c false.
        //----------------------------------------
        friend bool operator==(const ChunkInfoTimeScaleIterator& lhs, const ChunkInfoTimeScaleIterator& rhs);

        //========================================
        //! \brief Compare two iterators for inequality.
        //! \param[in] lhs  ChunkInfoTimeScaleIterator to compare.
        //! \param[in] rhs  ChunkInfoTimeScaleIterator to compare.
        //! \returns Either \c true if unequals or otherwise \c false.
        //----------------------------------------
        friend bool operator!=(const ChunkInfoTimeScaleIterator& lhs, const ChunkInfoTimeScaleIterator& rhs);

        //========================================
        //! \brief Get current reference to value of ChunkInfoTimeScale.
        //! \returns Reference to value of ChunkInfoTimeScale.
        //----------------------------------------
        ChunkInfoTimeScale::reference operator*();

        //========================================
        //! \brief Get current reference to value of ChunkInfoTimeScale.
        //! \returns Reference to value of ChunkInfoTimeScale.
        //----------------------------------------
        ChunkInfoTimeScale::const_reference operator*() const;

        //========================================
        //! \brief Get current pointer to value of ChunkInfoTimeScale.
        //! \returns Pointer to value of ChunkInfoTimeScale.
        //----------------------------------------
        ChunkInfoTimeScale::pointer operator->();

        //========================================
        //! \brief Get current pointer to value of ChunkInfoTimeScale.
        //! \returns Pointer to value of ChunkInfoTimeScale.
        //----------------------------------------
        ChunkInfoTimeScale::const_pointer operator->() const;

        //========================================
        //! \brief Iterate forward to next.
        //! \returns Reference to this.
        //----------------------------------------
        ChunkInfoTimeScaleIterator& operator++();

        //========================================
        //! \brief Iterate forward to next.
        //! \returns Reference to this before forward iteration.
        //----------------------------------------
        ChunkInfoTimeScaleIterator operator++(int);

        //========================================
        //! \brief Iterate backward to previous.
        //! \returns Reference to this.
        //----------------------------------------
        ChunkInfoTimeScaleIterator& operator--();

        //========================================
        //! \brief Iterate backward to previous.
        //! \returns Reference to this before backward iteration.
        //----------------------------------------
        ChunkInfoTimeScaleIterator operator--(int);

        //========================================
        //! \brief Get iterator on first element in map.
        //! \returns Iterator on first element.
        //----------------------------------------
        ChunkInfoTimeScaleIterator begin();

        //========================================
        //! \brief Get iterator on first element in map.
        //! \returns Iterator on first element.
        //----------------------------------------
        ChunkInfoTimeScaleIterator begin() const;

        //========================================
        //! \brief Get iterator on last element in map.
        //! \returns Iterator on last element.
        //----------------------------------------
        ChunkInfoTimeScaleIterator end();

        //========================================
        //! \brief Get iterator on last element in map.
        //! \returns Iterator on last element.
        //----------------------------------------
        ChunkInfoTimeScaleIterator end() const;

        //========================================
        //! \brief Get iterator which marks out of bounds position.
        //! \returns Iterator on out of bounds position.
        //----------------------------------------
        ChunkInfoTimeScaleIterator eof();

        //========================================
        //! \brief Get iterator which marks out of bounds position.
        //! \returns Iterator on out of bounds position.
        //----------------------------------------
        ChunkInfoTimeScaleIterator eof() const;

    private:
        //========================================
        //! \brief Map to iterate on.
        //----------------------------------------
        ChunkInfoTimeScale& m_timeScale;

        //========================================
        //! \brief Current iterator on map.
        //----------------------------------------
        ChunkInfoTimeScale::iterator m_current;

    }; // class ChunkInfoTimeScaleIterator

public:
    //========================================
    //! \brief Default constructor which requires an uri of format BAG.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit BagDataStreamReader(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagDataStreamReader(BagDataStreamReader&& reader);

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    BagDataStreamReader(const BagDataStreamReader&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagDataStreamReader() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    BagDataStreamReader& operator=(BagDataStreamReader&& reader);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagDataStreamReader& operator=(const BagDataStreamReader& reader) = delete;

public:
    //========================================
    //! \brief Get earliest time in stream.
    //! \note Before successful open the time will be epoch time (0).
    //! \returns Earliest time in stream.
    //----------------------------------------
    Ptp64Time getStartTime() const;

    //========================================
    //! \brief Get latest time in stream.
    //! \note Before successful open the time will be epoch time (0).
    //! \returns Latest time in stream.
    //----------------------------------------
    Ptp64Time getEndTime() const;

    //========================================
    //! \brief Set data filter on meta information.
    //! \note It is recommended to set the filter before open stream.
    //!       Otherwise parameter preFilterIfPossible has no effect.
    //! \param[in] filter               Filter function used on data meta information.
    //! \param[in] preFilterIfPossible  (optional) Either \c true if filter chunks at index times,
    //!                                 otherwise \c false. Default is \c true.
    //----------------------------------------
    void applyConnectionFilter(const ConnectionFilter& filter, bool preFilterIfPossible = true);

public: // implements StreamBase
    //========================================
    //! \brief Checks if the stream is not accessible or failed unrecoverable or EOF.
    //! \return Either \c true if the resource is in bad or EOF condition, otherwise \c false.
    //----------------------------------------
    bool isEof() const override;

    //========================================
    //! \brief Request resource access and takeover of the stream ownership.
    //! \note Supports only the uri protocols UriProtocol::File and UriProtocol::NoAddr.
    //! \param[in] ioStream  Resource Stream handle.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream) override;

    //========================================
    //! \brief Release resources and stream ownership.
    //! \returns Get stream ownership back.
    //----------------------------------------
    IoStreamPtr release() override;

    //========================================
    //! \brief Seek the cursor position.
    //! \param[in] cursor  Target cursor position.
    //! \return Either \c true if possible, otherwise \c false.
    //----------------------------------------
    bool seek(const int64_t cursor) override;

    //========================================
    //! \brief Seek cursor to begin of stream.
    //! \return Either \c true if succussful, otherwise \c false.
    //----------------------------------------
    bool seekBegin() override;

    //========================================
    //! \brief Seek cursor to end of stream.
    //! \return Either \c true if succussful, otherwise \c false.
    //----------------------------------------
    bool seekEnd() override;

public: // implements DataPackageStreamReader
    //========================================
    //! \brief Read first DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readFirstPackage() override;

    //========================================
    //! \brief Read last DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readLastPackage() override;

    //========================================
    //! \brief Read next DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readNextPackage() override;

    //========================================
    //! \brief Read previous DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readPreviousPackage() override;

    //========================================
    //! \brief Skip the next n DataPackage blocks.
    //! \param[in] packageCount  Count of packages too skip.
    //! \return Either \c true if succussfull, otherwise \c false.
    //----------------------------------------
    bool skipNextPackages(const uint32_t packageCount) override;

    //========================================
    //! \brief Skip the n previous DataPackage blocks.
    //! \param[in] packageCount  Count of packages too skip.
    //! \return Either \c true if succussfull, otherwise \c false.
    //----------------------------------------
    bool skipPreviousPackages(const uint32_t packageCount) override;

private:
    //========================================
    //! \brief Prepare BAG stream reading
    //!        by index chunks and connections.
    //! \returns Either \c true if indexing successful, otherwise \c false.
    //----------------------------------------
    bool indexChunksAndConnections();

    //========================================
    //! \brief Get chunk to read next data from.
    //! \returns Either \c iterator to read context of chunk, otherwise \c iterator of end().
    //----------------------------------------
    ChunkDataMap::iterator getNextChunk();

    //========================================
    //! \brief Get chunk to read previous data from.
    //! \returns Either \c iterator to read context of chunk, otherwise \c iterator of end().
    //----------------------------------------
    ChunkDataMap::iterator getPrevChunk();

    //========================================
    //! \brief Iterate data forward on chunk read context.
    //! \note Cean up chunk read context if eof in chunk.
    //! \param[in, out] chunk  Iterator on chunk read context.
    //----------------------------------------
    void moveForward(ChunkDataMap::iterator& chunk);

    //========================================
    //! \brief Iterate data backward on chunk read context.
    //! \note Cean up chunk read context if eof in chunk.
    //! \param[in, out] chunk  Iterator on chunk read context.
    //----------------------------------------
    void moveBackward(ChunkDataMap::iterator& chunk);

    //========================================
    //! \brief Create chunk read context by iterator on chunk info time scale.
    //! \param[in] chunkInfo  Iterator of chunk info from time scale.
    //! \returns Either \c iterator to read context of chunk, otherwise \c iterator of end().
    //----------------------------------------
    ChunkDataMap::iterator addChunkToRange(const ChunkInfoTimeScaleIterator& chunkInfo);

    //========================================
    //! \brief Read data from chunk.
    //! \param[in] chunk    Iterator of chunk read context.
    //! \param[in] data     Readed data.
    //! \returns Either \c true if successful read, otherwise \c false.
    //----------------------------------------
    bool readDataFromChunk(const ChunkDataMap::const_iterator& chunk, DataPackagePtr& data);

private:
    //========================================
    //! \brief Stream header info.
    //----------------------------------------
    HeaderInfo m_header;

    //========================================
    //! \brief Connections in stream indexed by connection id.
    //----------------------------------------
    ConnectionDataMap m_connections;

    //========================================
    //! \brief Open chunk read contexte.
    //----------------------------------------
    ChunkDataMap m_chunks;

    //========================================
    //! \brief Time indexed chunk infos.
    //----------------------------------------
    ChunkInfoTimeScale m_chunkInfoTimeScale;

    //========================================
    //! \brief Pre filter flag.
    //----------------------------------------
    bool m_preFilterIfPossible;

    //========================================
    //! \brief Filter function pointer.
    //----------------------------------------
    ConnectionFilter m_connectionFilter;

    //========================================
    //! \brief Time of current data.
    //! \note Used to add chunk read context if time overlapped.
    //----------------------------------------
    Ptp64Time m_currentTimeInStream;

    //========================================
    //! \brief Iterator to mark the lower boundary of cache.
    //----------------------------------------
    ChunkInfoTimeScaleIterator m_lowerBoundaryChunkInfo;

    //========================================
    //! \brief Iterator to mark the upper boundary of cache.
    //----------------------------------------
    ChunkInfoTimeScaleIterator m_highestBoundaryChunkInfo;

}; // class BagDataStreamReader

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
