//==============================================================================
//! \file
//!
//! \brief Implements basic write functionality for BAG file.
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
#include <microvision/common/sdk/io/bag/BagStreamBase.hpp>
#include <microvision/common/sdk/io/DataPackageStreamWriter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements basic write functionality for BAG file.
//! \extends BagStreamBase
//! \extends DataPackageStreamWriter
//------------------------------------------------------------------------------
class BagStreamWriterBase : public BagStreamBase, public virtual DataPackageStreamWriter
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using BagStreamBase::open;

    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using DataPackageStreamWriter::open;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagStreamWriterBase";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor which requires a Uri of format BAG.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit BagStreamWriterBase(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagStreamWriterBase(BagStreamWriterBase&& writer);

    //========================================
    //! \brief Copy constructor is disabled because of stream dependencie.
    //----------------------------------------
    BagStreamWriterBase(const BagStreamWriterBase&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagStreamWriterBase() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    BagStreamWriterBase& operator=(BagStreamWriterBase&& writer);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagStreamWriterBase& operator=(const BagStreamWriterBase& writer) = delete;

public: // implements DataPackageStreamWriter
    //========================================
    //! \brief Request resource access to append data if possible.
    //! \param[in] ioStream  Resource stream to move.
    //! \param[in] append    Either \c true if the resource requested is in append mode, otherwise \c false.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream, const bool appendMode) override;

public: // implements StreamBase
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

    //========================================
    //! \brief Get the current cursor position.
    //! \return Current cursor position or -1 for EOF.
    //----------------------------------------
    int64_t tell() override;

protected:
    //========================================
    //! \brief Request resource access with openmode for file stream.
    //! \param[in] openMode  Stream mode to open the file.
    //----------------------------------------
    bool openInternal(const std::ios_base::openmode openMode);

    //========================================
    //! \brief Write BAG record data package.
    //! \param[in] dataPackage  BagRecordPackage to write.
    //! \returns Either \c true if successful written, otherwise \c false.
    //----------------------------------------
    bool writeRecord(BagRecordPackage& dataPackage);

    //========================================
    //! \brief Write BAG record data package at stream position.
    //! \param[in] dataPackage  BagRecordPackage to write.
    //! \returns Either \c true if successful written, otherwise \c false.
    //----------------------------------------
    bool writeRecordAtPosition(BagRecordPackage& dataPackage);
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
