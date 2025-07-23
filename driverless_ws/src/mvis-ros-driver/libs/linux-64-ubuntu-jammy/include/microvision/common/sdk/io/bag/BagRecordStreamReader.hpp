//==============================================================================
//! \file
//!
//! \brief Implements functionallity to read BagRecordPackages from BAG stream.
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
#include <microvision/common/sdk/io/bag/BagStreamReaderBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements functionality to read BagRecordPackages from BAG stream.
//!
//! Provides functions to read forward in BAG stream.
//!
//! \note Backward reading is not supported by BAG format.
//! \extends BagStreamReaderBase
//------------------------------------------------------------------------------
class BagRecordStreamReader final : public BagStreamReaderBase
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using BagStreamReaderBase::open;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagRecordStreamReader";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor which requires a uri of format BAG.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit BagRecordStreamReader(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagRecordStreamReader(BagRecordStreamReader&& reader);

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    BagRecordStreamReader(const BagRecordStreamReader&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagRecordStreamReader() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    BagRecordStreamReader& operator=(BagRecordStreamReader&& reader);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagRecordStreamReader& operator=(const BagRecordStreamReader& reader) = delete;

public: // implements StreamBase
    //========================================
    //! \brief Request resource access and takeover of the stream ownership.
    //! \note Supports only the uri protocols UriProtocol::File and UriProtocol::NoAddr.
    //! \param[in] ioStream  Resource Stream handle.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream) override;

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

private: // implements BagStreamBase
    //========================================
    //! \brief Check whether path is valid format/schema.
    //! \return Either \c true if path format/schema is valid, otherwise \c false.
    //----------------------------------------
    bool checkPath() const override;

}; // class BagRecordStreamReader

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
