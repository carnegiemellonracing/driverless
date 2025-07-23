//==============================================================================
//! \file
//!
//! \brief Implements functionallity to write BagRecordPackages in BAG stream.
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
#include <microvision/common/sdk/io/bag/BagStreamWriterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements functionality to write BagRecordPackages in BAG stream.
//! \extends BagStreamWriterBase
//------------------------------------------------------------------------------
class BagRecordStreamWriter : public BagStreamWriterBase
{
private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagRecordStreamWriter";

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
    //! \brief Default constructor which requires a Uri of format BAG.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit BagRecordStreamWriter(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagRecordStreamWriter(BagRecordStreamWriter&& writer);

    //========================================
    //! \brief Copy constructor is disabled because of stream dependencie.
    //----------------------------------------
    BagRecordStreamWriter(const BagRecordStreamWriter&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagRecordStreamWriter() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    BagRecordStreamWriter& operator=(BagRecordStreamWriter&& writer);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagRecordStreamWriter& operator=(const BagRecordStreamWriter& writer) = delete;

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
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
