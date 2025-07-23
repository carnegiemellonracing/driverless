//==============================================================================
//! \file
//!
//! \brief Reader factory extension for BagStreamReader.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 03, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>

#include <microvision/common/sdk/extension/StreamReaderFactory.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Reader factory extension for BagStreamReader.
//------------------------------------------------------------------------------
class BagStreamReaderFactoryExtension : public StreamReaderFactoryExtension
{
public:
    //========================================
    //! \brief Default destructor
    //----------------------------------------
    ~BagStreamReaderFactoryExtension() override;

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagStreamReaderFactoryExtension";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Creates a DataPackageStreamReader from Uri.
    //! \param[in] path  Valid uri of source system.
    //! \return Either an instance of DataPackageStreamReader if supported or otherwise nullptr.
    //----------------------------------------
    DataPackageStreamReaderPtr createPackageReaderFromUri(const Uri& path) const override;
}; // BagStreamReaderFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
