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

#include <microvision/common/sdk/extension/ExporterFactoryExtension.hpp>
#include <microvision/common/sdk/extension/Extendable.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/logging/logging.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class SdkDataContainerExporterFactoryExtension;

//==============================================================================
//! \brief Factory extension to create plugin datatype exporters.
//!
//! This singleton factory is extendable and will create plugin datatype exporters.
//! See the demos and class ExporterBase for more information on idc data container export.
//------------------------------------------------------------------------------
class ExporterFactory final : public Extendable<ExporterFactoryExtension>
{
public:
    //========================================
    //! \brief ExporterBase shared pointer
    //!
    //! \note Do not confuse with the ExporterFactoryExtension::ExporterWeakPtr weak pointer!
    //----------------------------------------
    using ExporterPtr = std::shared_ptr<ExporterBase>;

public:
    //========================================
    //! \brief Copy constructor removed.
    //!
    //! No copies of the singleton.
    //----------------------------------------
    ExporterFactory(ExporterFactory const&) = delete;

    //========================================
    //! \brief Assignment copy construction removed.
    //!
    //! No copies of the singleton.
    //----------------------------------------
    void operator=(ExporterFactory const&) = delete;

    //========================================
    //! \brief Get the singleton instance.
    //!
    //! \return Plugin datatype exporter factory
    //----------------------------------------
    static ExporterFactory& getInstance();

public:
    //========================================
    //! \brief Get a data container exporter for a given data container identification to a serialized data type.
    //!
    //! \param[in] exporterId  Data container identification and serialized data type to get the exporter for.
    //! \return Exporter created by the factory.
    //!
    //! \note Exporters once created are cached and faster to get the next time.
    //----------------------------------------
    static ExporterPtr getExporter(const DataContainerBase::IdentificationKey& exporterId)
    {
        auto& factory = ExporterFactory::getInstance();
        if (factory.m_exporterMap[exporterId].expired())
        {
            factory.m_exporterMap[exporterId] = factory.createFromId(exporterId);
        }

        return factory.m_exporterMap[exporterId].lock();
    }

    //========================================
    //! \brief Get data container exporter which provides a binary serialization.
    //!
    //! \tparam ContainerType      The type of data container the exporter shall
    //!                            be able to serialize.
    //! \tparam serializationType  Data type ID of the serialization the exporter
    //!                            shall be able to write.
    //! \return A shared pointer to the exporter created by the factoy. If no
    //!         matching exporter has been registered, the shared pointer will
    //!         point to no instance (nullptr).
    //!
    //! \note Exporters once created are cached and faster to get the next time.
    //----------------------------------------
    template<typename ContainerType, DataTypeId::DataType serializationType>
    static ExporterPtr getExporter()
    {
        // create static instance of the id to keep
        static const DataContainerBase::IdentificationKey exporterRegId{
            serializationType, ContainerType::getClassIdHashStatic(), findUuid<ContainerType>()};

        return getExporter(exporterRegId);
    }

    //========================================
    //! \brief Get the exporter providing the default serialization for a given data container.
    //!
    //! Not all containers have a default exporter.
    //!
    //! \tparam ContainerType  Data container type for which the default exporter is wanted.
    //! \return Shared pointer to exporter if found. \c nullptr if not.
    //!
    //! \note This function is searching through all extensions. For reuse you should store the exporter.
    //----------------------------------------
    template<typename ContainerType>
    static ExporterPtr getDefaultExporter()
    {
        auto exporterCandidates = ExporterFactory::getInstance().findExporters(
            ContainerType::getClassIdHashStatic(), DataContainerBase::Uuid(), true);
        if (!exporterCandidates.empty())
        {
            return exporterCandidates.front().lock();
        }

        getLogger()->warning(LOGMSG << "No default exporter found for type " << ContainerType::getClassIdHashStatic()
                                    << "!");

        return ExporterPtr{};
    }

public:
    //========================================
    //! \brief Register this exporter factory to be used to export plugin data containers.
    //!
    //! \param[in] ext  The exporter factory extension.
    //! \return The exporter factory extension registered.
    //----------------------------------------
    const std::shared_ptr<ExporterFactoryExtension>
    registerExtension(const std::shared_ptr<ExporterFactoryExtension>& ext) override;

    //========================================
    //! \brief Remove this exporter factory from the registration.
    //!
    //! \param[in] ext  The exporter factory extension.
    //----------------------------------------
    void unregisterExtension(const std::shared_ptr<ExporterFactoryExtension>& ext) override;

private:
    //========================================
    //! \brief ExporterBase weak pointer
    //----------------------------------------
    using ExporterWeakPtr = std::weak_ptr<ExporterBase>;

    //========================================
    //! \brief Create datatype exporter for given exporter identification.
    //!
    //! \param[in] exporterId  Exported data container type and resulting data block serialization type id
    //!                        identifying the created exporter.
    //! \return Weak pointer to instance of exporter created by this factory.

    //----------------------------------------
    ExporterWeakPtr createFromId(const DataContainerBase::IdentificationKey& exporterId) const;

    //========================================
    //! \brief Search for all available exporters for the given ids.
    //!
    //! For each data container class there may or may not be one or several exporters available.
    //! Each exporter serializes the data container into another special data container binary format.
    //! Which one you use depends on your preferences and the origin of the data.
    //!
    //! \param[in] hashId                HashId of data container class which the exporter shall be able to export.
    //! \param[in] uuid                  Custom data container type unique id for wanted exporter.
    //! \param[in] defaultExportersOnly  Return only those found exporters that have been registered as default.
    //! \return All exporters found as weak pointers.
    //----------------------------------------
    std::vector<ExporterWeakPtr> findExporters(const DataContainerBase::HashId& hashId,
                                               const DataContainerBase::Uuid& uuid,
                                               const bool defaultExportersOnly = false);

    //========================================
    //! \brief Default constructor to be used only for singleton creation.
    //----------------------------------------
    ExporterFactory() = default;

    //========================================
    //! \brief Get logger of the ExporterFactory.
    //!
    //! \return Logger of the ExportFactory as shared pointer.
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr getLogger();

private:
    static constexpr const char* loggerId = "microvision::common::sdk::ExporterFactory";
    static microvision::common::logging::LoggerSPtr logger;

    static std::unique_ptr<ExporterFactory> factoryInstancePtr; // created only once

    std::unordered_map<DataContainerBase::IdentificationKey, ExporterWeakPtr> m_exporterMap; // created exporters cache
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
