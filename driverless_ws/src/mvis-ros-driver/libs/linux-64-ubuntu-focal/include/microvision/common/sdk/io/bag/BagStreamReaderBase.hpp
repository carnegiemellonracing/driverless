//==============================================================================
//! \file
//!
//! \brief Implements basic read functionallity for BAG file.
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
#include <microvision/common/sdk/io/DataPackageStreamReader.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements basic read functionality for BAG file.
//! \extends BagStreamBase
//! \extends DataPackageStreamReader
//------------------------------------------------------------------------------
class BagStreamReaderBase : public BagStreamBase, public virtual DataPackageStreamReader
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using BagStreamBase::open;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagStreamReaderBase";

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
    explicit BagStreamReaderBase(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagStreamReaderBase(BagStreamReaderBase&& reader);

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    BagStreamReaderBase(const BagStreamReaderBase&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagStreamReaderBase() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    BagStreamReaderBase& operator=(BagStreamReaderBase&& reader);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagStreamReaderBase& operator=(const BagStreamReaderBase& reader) = delete;

public: // implements StreamBase
    //========================================
    //! \brief Request resource access and takeover of the stream ownership.
    //! \note Supports only the uri protocols UriProtocol::File and UriProtocol::NoAddr.
    //! \param[in] ioStream  Resource Stream handle.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream) override;

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
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
