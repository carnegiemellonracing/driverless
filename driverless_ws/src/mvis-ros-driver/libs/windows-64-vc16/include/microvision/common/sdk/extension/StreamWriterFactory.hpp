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

#include <microvision/common/sdk/extension/StreamWriterFactoryExtension.hpp>
#include <microvision/common/sdk/io/IdcDataPackageStreamWriter.hpp>
#include <microvision/common/sdk/extension/Extendable.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Extension point to provide DataPackageStreamWriter implementations.
//!
//! This factory provides the extension point for DataPackageStreamWriter
//! implementations like IdcDataPackageStreamWriter to work with anonymous data sources.
//!
//! To implement a new data source add an implementation of
//! StreamWriterFactoryExtension on this factory.
//!
//! \example IdcStreamWriterFactoryExtension
//! \Note It is recommended to use the singleton instance by ::getInstance().
//------------------------------------------------------------------------------
class StreamWriterFactory final : public Extendable<StreamWriterFactoryExtension>
{
private:
    //========================================
    //! \brief Logger name for setup logger configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::StreamWriterFactory";

    //========================================
    //! \brief Provides common logger interface.
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

private:
    //========================================
    //! Constructor registering all MVIS SDK stream writer extensions.
    //!
    //! Writers which are not delivered as plugins to the customer have to be registered manually.
    //!
    //! \note When adding new writers with the sdk they have to be registered here.
    //----------------------------------------
    StreamWriterFactory();

public:
    //========================================
    //! \brief Get the singleton instance of StreamWriterFactory.
    //! \return Static instance of StreamWriterFactory.
    //----------------------------------------
    static StreamWriterFactory& getInstance();

public:
    //========================================
    //! \brief Create a DataPackageStreamWriter from destination file path.
    //!
    //! The format of the file will be identified by the file name extension.
    //!
    //! \param[in] filePath  Valid file path at destination system.
    //! \return Either an instance of DataPackageStreamWriter if supported or otherwise nullptr.
    //----------------------------------------
    DataPackageStreamWriterPtr createPackageWriterFromFile(const std::string& filePath) const;

    //========================================
    //! \brief Create a DataPackageStreamWriter from destination Uri.
    //!
    //! In general will the implementation selected
    //! either by the file name extension of the path in case of file
    //! or by Format or if ambiguous  as well by Schema and may Version.
    //!
    //! \param[in] path  Valid Uri at destination system.
    //! \return Either an instance of DataPackageStreamWriter if supported or otherwise nullptr.
    //----------------------------------------
    DataPackageStreamWriterPtr createPackageWriterFromUri(const Uri& path) const;

    //========================================
    //! \brief Create a IdcDataPackageStreamWriter from destination file path.
    //!
    //! The format of the file will be identified by the file name extension.
    //!
    //! \param[in] filePath  Valid file path at destination system.
    //! \return Either an instance of IdcDataPackageStreamWriter if supported or otherwise nullptr.
    //----------------------------------------
    IdcDataPackageStreamWriterPtr createIdcPackageWriterFromFile(const std::string& filePath) const;

    //========================================
    //! \brief Create a IdcDataPackageStreamWriter from destination Uri.
    //!
    //! In general will the implementation selected
    //! either by the file name extension of the path in case of file
    //! or by Format or if ambiguous  as well by Schema and may Version.
    //!
    //! \param[in] path  Valid Uri at destination system.
    //! \return Either an instance of IdcDataPackageStreamWriter if supported or otherwise nullptr.
    //----------------------------------------
    IdcDataPackageStreamWriterPtr createIdcPackageWriterFromUri(const Uri& path) const;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
