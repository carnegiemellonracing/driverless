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

#include <microvision/common/sdk/extension/StreamWriterFactory.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Writer factory extension for IdcStreamWriter.
//------------------------------------------------------------------------------
class IdcStreamWriterFactoryExtension : public StreamWriterFactoryExtension
{
public:
    ~IdcStreamWriterFactoryExtension() override = default;

private:
    //========================================
    //! \brief Static registration at linking time.
    //----------------------------------------
    static const typename StreamWriterFactory::TExtensionPtr m_instance;

    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::IdcStreamWriterFactoryExtension";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Creates a DataPackageStreamWriter from destination Uri.
    //! \param[in] path  Valid uri at destination system.
    //! \return Either an instance of DataPackageStreamWriter if supported or otherwise nullptr.
    //----------------------------------------
    DataPackageStreamWriterPtr createPackageWriterFromUri(const Uri& path) const override;
}; // IdcStreamWriterFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
