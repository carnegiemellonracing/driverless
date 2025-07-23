//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 5, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/StreamReaderFactory.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Reader factory extension for IdcStreamReader.
//------------------------------------------------------------------------------
class IdcStreamReaderFactoryExtension : public StreamReaderFactoryExtension
{
public:
    ~IdcStreamReaderFactoryExtension() override = default;

private:
    //========================================
    //! \brief Static registration at linking time.
    //----------------------------------------
    static const typename StreamReaderFactory::TExtensionPtr m_instance;

    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::IdcStreamReaderFactoryExtension";

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
}; // IdcStreamReaderFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
