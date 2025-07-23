//==============================================================================
//! \file
//!
//! \brief Implements functionality to write BagRecordPackages in BAG chunk record.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 08, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>
#include <microvision/common/sdk/io/bag/BagStreamWriterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements functionality to write BagRecordPackages in BAG chunk record.
//! \attention The chunk payload has to be resized before writing.
//! \extends BagStreamWriterBase
//------------------------------------------------------------------------------
class BagRecordChunkStreamWriter : public BagStreamWriterBase
{
private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagRecordChunkStreamWriter";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using BagStreamWriterBase::open;

public:
    //========================================
    //! \brief Default constructor which requires a chunk record package.
    //! \param[in] chunk  Chunk record.
    //----------------------------------------
    explicit BagRecordChunkStreamWriter(const BagChunkRecordPackage& chunk);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagRecordChunkStreamWriter(BagRecordChunkStreamWriter&& writer);

    //========================================
    //! \brief Copy constructor is disabled because of stream dependencie.
    //----------------------------------------
    BagRecordChunkStreamWriter(const BagRecordChunkStreamWriter&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagRecordChunkStreamWriter() override;

public:
    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    BagRecordChunkStreamWriter& operator=(BagRecordChunkStreamWriter&& writer) = delete;

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagRecordChunkStreamWriter& operator=(const BagRecordChunkStreamWriter& writer) = delete;

public: // implements DataPackageStreamWriter
    //========================================
    //! \brief Request resource access to append data if possible.
    //! \param[in] ioStream  Resource stream to move.
    //! \param[in] append    Disabled, \c true will cause an std::invalid_argument exception. Please use \c false.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream, const bool appendMode) override;

public: // implements DataPackageStreamWriter
    //========================================
    //! \brief Write DataPackage at stream position.
    //! \param[in/out] data  DataPackage to write.
    //! \returns Either \c true if DataPackage successful written, otherwise \c false.
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

private: // implements BagStreamBase
    //========================================
    //! \brief Check whether path is valid format/schema.
    //! \return Either \c true if path format/schema is valid, otherwise \c false.
    //----------------------------------------
    bool checkPath() const override;

private:
    //========================================
    //! \brief Chunk record.
    //----------------------------------------
    const BagChunkRecordPackage& m_chunk;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
