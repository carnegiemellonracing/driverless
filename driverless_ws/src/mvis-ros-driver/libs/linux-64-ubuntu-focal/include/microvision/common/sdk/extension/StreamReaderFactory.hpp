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

#include <microvision/common/sdk/extension/StreamReaderFactoryExtension.hpp>
#include <microvision/common/sdk/io/IdcDataPackageStreamReader.hpp>
#include <microvision/common/sdk/extension/Extendable.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Extension point to provide (Idc)DataPackageStreamReader implementations.
//!
//! This factory provides the extension point for DataPackageStreamReader
//! implementations like IdcStreamReader to work with anonymous data sources.
//! To implement a new data source add an implementation of StreamReaderFactoryExtension
//! on this factory.
//!
//! \example IdcStreamReaderFactoryExtension
//! \note It's recommended to use the singleton instance by ::getInstance().
//------------------------------------------------------------------------------
class StreamReaderFactory final : public Extendable<StreamReaderFactoryExtension>
{
private:
    //========================================
    //! \brief Logger name for setup logger configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::StreamReaderFactory";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

private:
    //========================================
    //! Constructor registering all MVIS SDK stream reader extensions.
    //!
    //! Readers which are not delivered as plugins to the customer have to be registered manually.
    //!
    //! \note When adding new readers with the sdk they have to be registered here.
    //----------------------------------------
    StreamReaderFactory();

public:
    //========================================
    //! \brief Get the singleton instance of StreamReaderFactory.
    //! \return Signleton instance of StreamReaderFactory.
    //----------------------------------------
    static StreamReaderFactory& getInstance();

public:
    //========================================
    //! \brief Create a DataPackageStreamReader from file path.
    //!
    //! The format of the file will identified by the file name extension.
    //!
    //! \param[in] filePath  Valid file path at source system.
    //! \return Either DataPackageStreamReader instance if supported, otherwise nullptr.
    //----------------------------------------
    DataPackageStreamReaderPtr createPackageReaderFromFile(const std::string& filePath) const;

    //========================================
    //! \brief Create a DataPackageStreamReader from source Uri.
    //!
    //! In general will the implementation selected
    //! either by the file name extension of the path in case of file
    //! or by Format or if ambiguous  as well by Schema and may Version.
    //!
    //! \param[in] path  Valid Uri path at source system.
    //! \return Either DataPackageStreamReader instance if supported, otherwise nullptr.
    //----------------------------------------
    DataPackageStreamReaderPtr createPackageReaderFromUri(const Uri& args) const;

    //========================================
    //! \brief Create a IdcDataPackageStreamReader from file path.
    //!
    //! The format of the file will identified by the file name extension.
    //!
    //! \param[in] filePath  Valid file path at source system.
    //! \return Either IdcDataPackageStreamReader instance if supported, otherwise nullptr.
    //----------------------------------------
    IdcDataPackageStreamReaderPtr createIdcPackageReaderFromFile(const std::string& filePath) const;

    //========================================
    //! \brief Create a IdcDataPackageStreamReader from source Uri.
    //!
    //! In general will the implementation selected
    //! either by the file name extension of the path in case of file
    //! or by Format or if ambiguous  as well by Schema and may Version.
    //!
    //! \param[in] path  Valid Uri path at source system.
    //! \return Either IdcDataPackageStreamReader instance if supported, otherwise nullptr.
    //----------------------------------------
    IdcDataPackageStreamReaderPtr createIdcPackageReaderFromUri(const Uri& args) const;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
