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
//! \date Sep 5th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/ImporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface for the plugin datatype exporter extension.
//!
//! This interface will create plugin datatype exporters.
//------------------------------------------------------------------------------
class ExporterFactoryExtension
{
    friend class ExporterFactory;

public:
    virtual ~ExporterFactoryExtension() = default;

public:
    //========================================
    //! \brief Exporter pointer type.
    //----------------------------------------
    using ExporterWeakPointer = std::weak_ptr<ExporterBase>;

public:
    //========================================
    //! \brief Get all the data container identifications for which this factory extension can create exporters.
    //!
    //! \return List all the data container identifications for which this factory extension provides exporters.
    //!
    //! \note One extension can hold multiple importers.
    //----------------------------------------
    virtual std::vector<DataContainerBase::IdentificationKey> getIds() const = 0;

    //========================================
    //! \brief Create an exporter plugin for the given data container.
    //!
    //! \param[in] dataContainerId  Data container identification for which this extension provides an exporter.
    //! \return Exporter created by this factory extension.
    //!
    //! \note A weak pointer is returned here to enable automatic plugin unregistration of exporters for the factory.
    //!       Your derived extension needs to create and hold a shared pointer and then return a weak pointer here.
    //!		  See also \sa TestFactoryExtension in ExtensionTests.
    //----------------------------------------
    virtual ExporterWeakPointer createFromId(const DataContainerBase::IdentificationKey& dataContainerId) = 0;

protected:
    //========================================
    //! \brief Check if this extension provides a default serialization for the given data container type
    //!        identification (like Scan -> Scan2209).
    //!
    //! This function is used to identify default exporters in the SdkDataContainerFactoryExtension
    //! but can be overridden to add new defaults from plugin extensions.
    //!
    //! \param[in] dataContainerIdentifier  The data container identifier for which this exporter factory extension
    //!                                     might provide the default serialization (a binary serialized data type).
    //! \return \c True if the \a dataContainerIdentifier is associated with a default exporter in this extension,
    //!         \c false if the associated exporter is not the default exporter for the associated data container
    //!         or if not registered here.
    //----------------------------------------
    virtual bool isDefaultExporterFor(const DataContainerBase::IdentificationKey& dataContainerIdentifier)
    {
        (void)dataContainerIdentifier;
        return false; // no default exporter if not overridden
    }
}; // ExporterFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
