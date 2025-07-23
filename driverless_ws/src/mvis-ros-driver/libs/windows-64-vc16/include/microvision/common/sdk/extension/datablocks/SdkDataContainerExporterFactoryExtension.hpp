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
//! \date Dec 18th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/extension/ExporterFactoryExtension.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Exporter extension for sdk standard data container type exporters.
//!
//! All standard sdk data container types are registered here with all available exporters.
//!
//! \note This is used internally by the sdk to register its data containers. Do not use elsewhere.
//------------------------------------------------------------------------------
class SdkDataContainerExporterFactoryExtension final : public ExporterFactoryExtension
{
    friend class ExporterFactory;

private:
    using ExporterCreateFunction
        = std::function<std::shared_ptr<ExporterBase>(void)>; // function type to create a new exporter

    // storage container for all registered exporters
    struct ExporterEntry
    {
        ExporterCreateFunction createFunction;
        std::shared_ptr<ExporterBase> exporterInstancePtr; // once created exporter
        bool isDefault; // default flag
    };

private:
    //===============================================================
    //! \brief Constructor - register all idc data containers.
    //!
    //! All exporters for special and general types of data containers are registered here
    //! so they can be found to create serialized binary data blocks from data containers.
    //!
    //! \note All sdk supplied idc data container exporters are registered here!
    //---------------------------------------------------------------
    SdkDataContainerExporterFactoryExtension();

public:
    //===============================================================
    //! \brief Default destructor.
    //---------------------------------------------------------------
    ~SdkDataContainerExporterFactoryExtension() override = default;

public:
    //===============================================================
    //! \brief Get all data container identifications for which this extension provides exporters.
    //!
    //! An extension can contain more than one exporter.
    //!
    //! \note All sdk supplied idc data container exporters are returned here!
    //!
    //! \return List all the data container identifications for which this factory extension provides exporters.
    //---------------------------------------------------------------
    std::vector<DataContainerBase::IdentificationKey> getIds() const override;

    //===============================================================
    //! \brief Create an exporter plugin from arguments.
    //!
    //! \param[in] id  Id for creating the exporter.
    //! \return Exporter created by this factory.
    //---------------------------------------------------------------
    std::weak_ptr<ExporterBase> createFromId(const DataContainerBase::IdentificationKey& id) override;

public: // idc data container exporters registration
    //===============================================================
    //! \brief Register an idc data container exporter for creation by this extension.
    //!
    //! \tparam ContainerType  Type of exporter to be registered.
    //! \tparam id             idc dataType id for the binary serialization this exporter shall serialize into.
    //---------------------------------------------------------------
    template<typename ContainerType, microvision::common::sdk::DataTypeId::DataType id>
    void registerExporter()
    {
        const DataContainerBase::IdentificationKey regId{
            id, ContainerType::getClassIdHashStatic(), findUuid<ContainerType>()};

        if (m_exporterMap.find(regId) != m_exporterMap.end())
        {
            throw std::invalid_argument("exporter already registered for id");
        }

        ExporterEntry newEntry = {[]() { return std::make_shared<Exporter<ContainerType, id>>(); }, nullptr, false};
        m_exporterMap[regId]   = newEntry;
    }

    //===============================================================
    //! \brief Register a default idc data container exporter for creation by this extension.
    //!
    //! A default exporter can only be registered for general data containers.
    //!
    //! \tparam ContainerType  Type of exporter to be registered.
    //! \tparam id             idc dataType id for the binary serialization this exporter shall serialize into.
    //---------------------------------------------------------------
    template<typename ContainerType,
             microvision::common::sdk::DataTypeId::DataType id,
             typename = typename std::enable_if<!std::is_base_of<SpecializedDataContainer, ContainerType>::value>::type>
    void registerDefaultExporter()
    {
        const DataContainerBase::IdentificationKey regId{
            id, ContainerType::getClassIdHashStatic(), findUuid<ContainerType>()};

        if (m_exporterMap.find(regId) != m_exporterMap.end())
        {
            throw std::invalid_argument("default exporter already registered for id");
        }

        ExporterEntry newEntry = {[]() { return std::make_shared<Exporter<ContainerType, id>>(); }, nullptr, true};
        m_exporterMap[regId]   = newEntry;
    }

protected:
    //===============================================================
    //! \brief Check if the extension provides a default exporter for the given special data container id.
    //!
    //! For a general data container the default exporter is that exporter that will be used when serializing
    //! data when non other is given explicitly. Normally the default exporter is that exporter which performs
    //! the serialization which is matching best to minimize data loss due to serialization.
    //!
    //! \note There may be cases, where other available exporters are to be preferred.
    //!
    //! \param[in] exporterIdentifier  The identifier for the exporter, gives information about the serialization,
    //!                                the container to be used and an additional uuid.
    //! \return \c True if the \a exporterIdentifier is associated to a default exporter, \c false if
    //!         the associated exporter is not the default exporter for the associated data container or if not
    //!         registered.
    //---------------------------------------------------------------
    bool isDefaultExporterFor(const DataContainerBase::IdentificationKey& exporterIdentifier) override
    {
        auto entryIterator = m_exporterMap.find(exporterIdentifier);
        if (entryIterator == m_exporterMap.end())
        {
            return false;
        }

        return entryIterator->second.isDefault;
    }

private:
    std::unordered_map<DataContainerBase::IdentificationKey, ExporterEntry>
        m_exporterMap; // stored registered, possibly already created exporters
}; // namespace sdk

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
