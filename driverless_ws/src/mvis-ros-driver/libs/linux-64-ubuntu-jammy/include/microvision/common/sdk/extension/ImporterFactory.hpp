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

#include <microvision/common/sdk/extension/ImporterFactoryExtension.hpp>
#include <microvision/common/sdk/extension/Extendable.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/logging/logging.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class SdkDataContainerImporterFactoryExtension;

//==============================================================================
//! \brief Factory extension to create plugin datatype (IdcDataContainer) importers
//!
//! This singleton factory is extendable and will create plugin datatype importers
//! See the demos and class ImporterBase for more information on idc data container import.
//------------------------------------------------------------------------------
class ImporterFactory final : public Extendable<ImporterFactoryExtension>
{
public:
    //========================================
    //! \brief ImporterBase shared pointer type
    //!
    //! \note Do not confuse with the ImporterFactoryExtension::ImporterPtr weak pointer!
    //----------------------------------------
    using ImporterPtr = std::shared_ptr<ImporterBase>;

public:
    //========================================
    //! \brief Copy constructor removed.
    //!
    //! No copies of the singleton.
    //----------------------------------------
    ImporterFactory(ImporterFactory const&) = delete;

    //========================================
    //! \brief Assignment copy construction removed.
    //!
    //! No copies of the singleton.
    //----------------------------------------
    void operator=(ImporterFactory const&) = delete;

    //========================================
    //! \brief Get the singleton instance.
    //!
    //! \return Plugin datatype importer factory
    //----------------------------------------
    static ImporterFactory& getInstance();

public:
    //========================================
    //! \brief Try to get plugin datatype importer from data type identification.
    //!
    //! \param[in] importerId  Importer identification with target data container type
    //!                        and data type id data block to be deserialized.
    //! \return Shared pointer to importer created by the factory.
    //!
    //! \note Importers once created are cached and faster to get the next time.
    //----------------------------------------
    static ImporterPtr getImporter(const DataContainerBase::IdentificationKey& importerId);

    //========================================
    //! \brief Get datatype importer.
    //!
    //! \tparam ImporterType  Type of data container importer wanted.
    //! \return Weak pointer to importer. \c nullptr if not found.
    //!
    //! \note Importers once created are cached and faster to get the next time.
    //----------------------------------------
    template<typename ImporterType>
    static ImporterPtr getImporter()
    {
        using ContainerType = typename std::remove_reference<decltype(*ImporterType::createContainerStatic())>::type;

        // create static instance of the id to keep
        static const DataContainerBase::IdentificationKey importerId{
            ImporterType::getDataTypeStatic(), ContainerType::getClassIdHashStatic(), findUuid<ContainerType>()};

        return getImporter(importerId);
    } // namespace sdk

    //========================================
    //! \brief Try to get general data container importer from data type.
    //!
    //! \tparam ContainerType      General data container to import to from given data type.
    //! \param[in] serializedType  Data type ID of the data block to be imported.
    //! \return Importer created by the factory.
    //!
    //! \note Importers once created are cached and faster to get the next time.
    //----------------------------------------
    template<typename ContainerType,
             typename = typename std::enable_if<!std::is_base_of<SpecializedDataContainer, ContainerType>::value>::type>
    static ImporterPtr getGeneralImporter(const DataTypeId::DataType serializedType)
    {
        const DataContainerBase::IdentificationKey importerId{
            serializedType, ContainerType::getClassIdHashStatic(), findUuid<ContainerType>()};

        return getImporter(importerId);
    }

    //========================================
    //! \brief Get general data container type importer for a serialized data block special data type.
    //!
    //! A general importer does not exist for all data types. For example the general data container types themselves
    //! have no serialized form (hence getDefaultExporter exists) so they also have no general importer.
    //!
    //! \param[in] serializedDataTypeId  Identification of serialized data block type to find an importer
    //!                                  that imports into a general container.
    //! \return Shared pointer to importer found, empty if not.
    //!
    //! \note This is searching though all importer extensions. Store the importer for reuse.
    //----------------------------------------
    static ImporterPtr getGeneralImporter(const DataTypeId& serializedDataTypeId);

public:
    //========================================
    //! \brief Register this importer factory extension.
    //!
    //! \param[in] ext  The importer factory extension.
    //! \return The importer factory extension registered.
    //----------------------------------------
    const std::shared_ptr<ImporterFactoryExtension>
    registerExtension(const std::shared_ptr<ImporterFactoryExtension>& ext) override;

    //========================================
    //! \brief Remove this importer factory extension.
    //!
    //! \param[in] ext  The importer factory extension.
    //----------------------------------------
    void unregisterExtension(const std::shared_ptr<ImporterFactoryExtension>& ext) override;

private:
    //========================================
    //! \brief ImporterBase weak pointer type
    //----------------------------------------
    using ImporterWeakPtr = std::weak_ptr<ImporterBase>;

    //========================================
    //! \brief Default constructor to be used only for singleton creation.
    //----------------------------------------
    ImporterFactory() = default;

    //========================================
    //! \brief Create plugin datatype importer from serialized data block identification.
    //!
    //! \param[in] serializedType  Data type identification to create the importer.
    //! \return Importer created by the factory.
    //----------------------------------------
    ImporterWeakPtr createFromId(const DataContainerBase::IdentificationKey& serializedType) const;

private:
    static constexpr const char* loggerId = "microvision::common::sdk::ImporterFactory";
    static microvision::common::logging::LoggerSPtr logger;

    static std::unique_ptr<ImporterFactory> factoryInstancePtr;

    ThreadSafe<std::unordered_map<DataContainerBase::IdentificationKey, ImporterWeakPtr>>
        m_importerMap; // created importers cache
};

//===============================================================================
//! \brief Helper for easier custom importer registration.
//!
//! This hides the extension for the user code.
//!
//! \tparam ImporterType  Type of Importer in user code.
//-------------------------------------------------------------------------------
template<typename ImporterType>
class RegisteredImporterFactoryExtension : public ImporterFactoryExtension
{
public:
    //========================================
    //! \brief Constructor
    //!
    //! \param[in] serializedType  Complete id of class for which an importer is to be found.
    //----------------------------------------
    explicit RegisteredImporterFactoryExtension(const DataContainerBase::IdentificationKey& serializedType)
      : ImporterFactoryExtension(), m_importerId{serializedType}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~RegisteredImporterFactoryExtension() override = default;

public:
    //========================================
    //! \brief Return the data type identification of the importers that this factory can create.
    //!
    //! \return List of data type ids deserialized by importers that this factory can create.
    //----------------------------------------
    std::vector<DataContainerBase::IdentificationKey> getIds() const override { return {m_importerId}; }

    //========================================
    //! \brief Create an importer plugin from data container type id.
    //!
    //! \param[in] importerId  Imported data block data type id and target data container type
    //!                        which identify the created importer.
    //! \return Shared pointer to instance of importer created by this factory.
    //----------------------------------------
    ImporterPtr createFromId(const DataContainerBase::IdentificationKey& importerId) override
    {
        (void)importerId;
        if (!m_importerPtr)
        {
            m_importerPtr = std::make_shared<ImporterType>();
        }

        return m_importerPtr;
    }

private:
    DataContainerBase::IdentificationKey
        m_importerId; // the serialized data type id and target data container combination this holds an importer for
    std::shared_ptr<ImporterType> m_importerPtr; // importer holder after first creation
};

//===============================================================================
//! \brief Register a custom importer in a simple way.
//!
//! This hides the extension for the user code by creating a simple extension containing only the
//! creation method for this one data container type.
//!
//! \note Use this method to register custom data container type importers.
//!
//! \tparam ImporterType  Type of importer in user code.
//! \param[in] regId      Data type and container identification used to register the importer type.
//! \return Shared pointer to the created extension. Use this to unregister when your plugin is shut down.
//-------------------------------------------------------------------------------
template<typename ImporterType>
std::shared_ptr<RegisteredImporterFactoryExtension<ImporterType>>
registerImporter(const DataContainerBase::IdentificationKey& regId)
{
    auto extensionPtr = std::make_shared<RegisteredImporterFactoryExtension<ImporterType>>(regId);
    ImporterFactory::getInstance().registerExtension(extensionPtr);

    return extensionPtr;
}

//===============================================================================
//! \brief Register a custom importer in a simple way.
//!
//! This hides the extension for the user code by creating a simple extension containing only the
//! creation method for this one data container type.
//!
//! \tparam ImporterType  Type of importer in user code.
//! \return Shared pointer to the created extension. Use this to unregister when your plugin is shut down.
//-------------------------------------------------------------------------------
template<typename ImporterType,
         typename ContainerType
         = typename std::remove_reference<decltype(*ImporterType::createContainerStatic())>::type>
std::shared_ptr<RegisteredImporterFactoryExtension<ImporterType>> registerImporter()
{
    // create static instance of the id to keep
    static const DataContainerBase::IdentificationKey regId{
        ImporterType::getDataTypeStatic(), ContainerType::getClassIdHashStatic(), findUuid<ContainerType>()};

    return registerImporter<ImporterType>(regId);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
