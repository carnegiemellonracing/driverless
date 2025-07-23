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
#include <microvision/common/sdk/extension/ImporterFactoryExtension.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//===============================================================
//! \brief Importer extension for sdk standard data container type importers.
//!
//! All standard sdk data container types are registered here with all available importers.
//!
//! \note This is used internally by the sdk to register its data containers. Do not use elsewhere.
//---------------------------------------------------------------
class SdkDataContainerImporterFactoryExtension final : public ImporterFactoryExtension
{
    friend class ImporterFactory;

private:
    using ImporterCreateFunction = std::function<std::shared_ptr<ImporterBase>(void)>;

    // used to store registered importers
    struct ImporterEntry
    {
        ImporterCreateFunction createFunction; // function to create importer when used first time
        std::shared_ptr<ImporterBase> importerInstancePtr; // pointer to created importer after first creation
        bool isGeneral; // general importer flag
    };

private:
    //===============================================================
    //! \brief Constructor - register all idc data containers.
    //!
    //! All importers for special and general types of data containers are registered here
    //! so they can be found for import of serialized binary data packages.
    //!
    //! \note All sdk supplied idc data container importers are registered here!
    //---------------------------------------------------------------
    SdkDataContainerImporterFactoryExtension();

public:
    //===============================================================
    //! \brief Default destructor.
    //---------------------------------------------------------------
    ~SdkDataContainerImporterFactoryExtension() override = default;

public:
    //===============================================================
    //! \brief Get all data container identifications for which this extension provides importers.
    //!
    //! An extension can contain more than one importer.
    //!
    //! \note All sdk supplied idc data container importers are returned here!
    //!
    //! \return List all the data container identifications for which this factory extension provides importers.
    //---------------------------------------------------------------
    std::vector<DataContainerBase::IdentificationKey> getIds() const override;

    //===============================================================
    //! \brief Create an importer for a data container.
    //!
    //! \param[in] id  Data container identification for creating the importer.
    //! \return Importer created by this factory.
    //---------------------------------------------------------------
    ImporterPtr createFromId(const DataContainerBase::IdentificationKey& id) override;

public: // idc data container importers registration
    //===============================================================
    //! \brief Register an idc data container importer for creation by this extension.
    //!
    //! \tparam ImporterType  Type of importer to be registered.
    //---------------------------------------------------------------
    template<
        typename ImporterType,
        typename ContainerType = typename std::remove_reference<decltype(*ImporterType::createContainerStatic())>::type,
        typename IsGeneral = typename std::conditional<std::is_base_of<SpecializedDataContainer, ContainerType>::value,
                                                       std::false_type,
                                                       std::true_type>::type>
    void registerImporter()
    {
        const DataContainerBase::IdentificationKey id{
            ImporterType::getDataTypeStatic(), ContainerType::getClassIdHashStatic(), findUuid<ContainerType>()};

        if (m_importerMap.find(id) != m_importerMap.end())
        {
            throw std::invalid_argument("importer already registered for id");
        }

        ImporterEntry newEntry = {ImporterType::createImporter, nullptr, IsGeneral::value};
        m_importerMap[id]      = newEntry;
    }

protected:
    //===============================================================
    //! \brief Check if the importer provides a general importer for the given special data container id.
    //!
    //! A general importer is used to import from a binary serialized \c special data container (data block)
    //! into a general data container.
    //!
    //! \note A general importer may \c not exist for all types!
    //!
    //! \param[in] importerIdentifier  The identifier for the exporter, gives information about the serialization,
    //!                                the container to be used and an additional uuid.
    //! \return \c True if the \a exporterIdentifier is associated to a default exporter, \c false if
    //!         the associated exporter is not the default exporter for the associated data container or if not
    //!         registered.
    //---------------------------------------------------------------
    bool isGeneralImporterFor(const DataContainerBase::IdentificationKey& importerIdentifier) override
    {
        auto entryIterator = m_importerMap.find(importerIdentifier);
        if (entryIterator == m_importerMap.end())
        {
            return false;
        }

        return entryIterator->second.isGeneral;
    }

private:
    std::unordered_map<DataContainerBase::IdentificationKey, ImporterEntry> m_importerMap; // all registered importers
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
