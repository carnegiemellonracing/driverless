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

#include <microvision/common/sdk/datablocks/ImporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface for the plugin datatype importer extension
//!
//! This interface will create plugin datatype importers.
//!
//! \note Derive your own importer extension from this. Register your extension at the ImporterFactory.
//!       Unregister when the plugin is shut down.
//!       For simple one importer extension see \sa RegisteredImporterFactoryExtension which provides
//!       an easy method to just register one importer.
//------------------------------------------------------------------------------
class ImporterFactoryExtension
{
    friend class ImporterFactory;

public:
    virtual ~ImporterFactoryExtension() = default;

public:
    //========================================
    //! \brief Importer pointer type
    //----------------------------------------
    using ImporterPtr = std::weak_ptr<ImporterBase>;

public:
    //========================================
    //! \brief Get all the serialized data type identifications for which this factory extension can create importers.
    //!
    //! \return List all the data container serialization identifications of importers that this factory extension provides.
    //!
    //! \note One extension can hold multiple importers.
    //----------------------------------------
    virtual std::vector<DataContainerBase::IdentificationKey> getIds() const = 0;

    //========================================
    //! \brief Create an importer for a specified serialized data type.
    //!
    //! \param[in] serializedType  Identification of serialized data type for creating the importer.
    //! \return Weak pointer to the importer created by this factory extension. Empty weak pointer if not possible.
    //!
    //! \note A weak pointer is returned here to enable automatic plugin unregistration of importers for the factory.
    //!       Your derived extension needs to create and hold a shared pointer and then return a weak pointer here.
    //!          See also \sa RegisteredImporterFactoryExtension and \sa TestFactoryExtension in ExtensionTests.
    //----------------------------------------
    virtual ImporterPtr createFromId(const DataContainerBase::IdentificationKey& serializedType) = 0;

protected:
    //========================================
    //! \brief Check if the given serialized data type has been registered as an importer to the general data
    //!        container type (like Scan2209 -> Scan).
    //!
    //! This function is used to identify general importers in the SdkDataContainerFactoryExtension
    //! but can be overridden to add new general importers from plugin extensions.
    //!
    //! \param[in] importerIdentifier  The identifier for the exporter, gives information about the serialization,
    //!                                the container to be used and an additional uuid.
    //! \return \c True if the \a importerIdentifier is associated to a general importer, \c false if
    //!         the associated importer is not a general importer for the associated data container or if not
    //!         registered.
    //----------------------------------------
    virtual bool isGeneralImporterFor(const DataContainerBase::IdentificationKey& importerIdentifier)
    {
        (void)importerIdentifier;
        return false; // no general importer if not overridden
    }
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
