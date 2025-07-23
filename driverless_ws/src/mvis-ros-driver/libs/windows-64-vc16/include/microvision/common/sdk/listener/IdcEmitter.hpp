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
//! \date Feb 10, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/sdk/listener/Emitter.hpp>
#include <microvision/common/sdk/config/Configuration.hpp>
#include <microvision/common/sdk/listener/DataContainerListener.hpp>
#include <microvision/common/sdk/listener/IdcDataPackageListener.hpp>

#include <microvision/common/logging/logging.hpp>

#include <unordered_map>
#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface which provides the functionality for idc data container/package listeners.
//!
//! Specialized for idc data. Derived virtual to ensure member ambiguity for Emitter.
//!
//! \sa IdcDataPackageListener
//------------------------------------------------------------------------------
class IdcEmitter : public virtual Emitter
{
public:
    //========================================
    //! \brief ImporterBase shared pointer type
    //!
    //! \note Do not confuse with the ImporterFactoryExtension::ImporterPtr weak pointer!
    //----------------------------------------
    using ImporterPtr = std::shared_ptr<ImporterBase>;

    //========================================
    //! \brief List of Configuration pointers.
    //----------------------------------------
    using ImporterConfigurationList = std::list<ConfigurationPtr>;

    //========================================
    //! \brief Map of configuration pointers indexed by configuration type.
    //----------------------------------------
    using ImporterConfigurationMapByType = std::unordered_map<std::string, ImporterConfigurationList>;

    //========================================
    //! \brief Nullable DataContainerListenerBase pointer.
    //----------------------------------------
    using DataContainerListenerPtr = std::shared_ptr<DataContainerListenerBase>;

    //========================================
    //! \brief Nullable DataContainerListenerBase pointer.
    //----------------------------------------
    using IdcDataPackageListenerPtr = std::shared_ptr<IdcDataPackageListener>;

    //========================================
    //! \brief Context of deserializing data package.
    //----------------------------------------
    struct DataContainerImporterContext
    {
    public:
        ImporterPtr importerPtr{}; //!< Importer which used to deserialize data package.
        ConfigurationPtr configurationPtr{}; //!< Configuration which used for deserialization.
        DataContainerPtr dataContainerPtr{}; //!< Data container which deserialized by importer.
    };

    //========================================
    //! \brief List of deserializing data package contexts.
    //----------------------------------------
    using DataContainerImporterContextList = std::list<DataContainerImporterContext>;

    //========================================
    //! \brief List of deserializing data package contexts.
    //----------------------------------------
    using IdcDataPackageListenerList = std::list<IdcDataPackageListenerPtr>;

    //========================================
    //! \brief List of data container listener pointers.
    //----------------------------------------
    using DataContainerListenerList = std::list<DataContainerListenerPtr>;

    //========================================
    //! \brief Map of data container listener pointers indexed by data container identification key.
    //----------------------------------------
    using DataContainerListenerMapByContainerId
        = std::unordered_map<DataContainerBase::IdentificationKey, DataContainerListenerList>;

    //========================================
    //! \brief Map of data container listener shared pointers associated to data type and container identification key.
    //----------------------------------------
    using DataContainerListenerMapByDataType = std::unordered_map<DataTypeId, DataContainerListenerMapByContainerId>;

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::IdcEmitter";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IdcEmitter();

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    IdcEmitter(const IdcEmitter& other);

    //========================================
    //! \brief Disabled move constructor.
    //----------------------------------------
    IdcEmitter(IdcEmitter&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcEmitter() override = default;

public:
    //========================================
    //! \brief Register listener at entity to receive idc data packages.
    //! \param[in] listener  Pointer to idc data package listener.
    //----------------------------------------
    void registerIdcDataPackageListener(const IdcDataPackageListenerPtr& listener);

    //========================================
    //! \brief Unregister listener from entity to avoid receiving idc data packages.
    //! \param[in] listener  Pointer to idc data package listener.
    //----------------------------------------
    void unregisterIdcDataPackageListener(const IdcDataPackageListenerPtr& listener);

public:
    //========================================
    //! \brief Register listener at entity to receive data containers.
    //! \param[in] listener  Pointer to data container listener.
    //----------------------------------------
    void registerDataContainerListener(const DataContainerListenerPtr& listener);

    //========================================
    //! \brief Unregister listener from entity to avoid receiving data containers.
    //! \param[in] listener  Pointer to data container listener.
    //----------------------------------------
    void unregisterDataContainerListener(const DataContainerListenerPtr& listener);

public:
    //========================================
    //! \brief Notify idc data package listeners about an idc data package which has been received.
    //! \param[in] dataPackage  Shared pointer to instance of idc data package about which the listener will be notified.
    //! \returns Either \c true if listeners are triggered or otherwise \c false.
    //----------------------------------------
    bool notifyIdcDataPackageListeners(const IdcDataPackagePtr& dataPackage);

    //========================================
    //! \brief Notify data container listeners about the idc data package which has been received.
    //! \param[in] dataPackage  Shared pointer to instance of idc data package about which the listener will be notified.
    //! \returns Either \c true if listeners are triggered or otherwise \c false.
    //----------------------------------------
    bool notifyDataContainerListeners(const IdcDataPackagePtr& dataPackage);

    //========================================
    //! \brief Notify data container listeners about imported data packages.
    //! \param[in] dataContainers  Contexts of imported data packages.
    //! \returns Either \c true if listeners are triggered or otherwise \c false.
    //----------------------------------------
    bool notifyDataContainerListeners(const DataContainerImporterContextList& dataContainers);

    //========================================
    //! \brief Notify data container listeners about imported data package.
    //! \param[in] dataContainer    Data container which deserialized by importer.
    //! \param[in] importer         Importer which used to deserialize data package.
    //! \param[in] configuration    Configuration which used for deserialization.
    //! \returns Either \c true if listeners are triggered or otherwise \c false.
    //----------------------------------------
    bool notifyDataContainerListeners(const DataContainerPtr& dataContainer,
                                      const ImporterPtr& importer,
                                      const ConfigurationPtr& configuration = nullptr);

public:
    //========================================
    //! \brief Get registered importer configurations of configuration type.
    //!
    //! The registered data container listener will trigger a deserialization of data package.
    //! And for that a configuration can be used to interpret the data during deserialization in different ways.
    //! The deserialization will be done for each registered configuration which is supported by importer.
    //!
    //! \param[in] configurationType  Configuration type.
    //! \returns Registered importer configurations.
    //----------------------------------------
    ImporterConfigurationList getImporterConfigurations(const std::string& configurationType) const;

    //========================================
    //! \brief Register importer configuration.
    //!
    //! The registered data container listener will trigger a deserialization of data package.
    //! And for that a configuration can be used to interpret the data during deserialization in different ways.
    //! The deserialization will be done for each registered configuration which is supported by importer.
    //!
    //! \param[in] importerConfig  Configuration pointer.
    //----------------------------------------
    void registerImporterConfiguration(const ConfigurationPtr& importerConfig);

    //========================================
    //! \brief Unregister importer configuration.
    //!
    //! This configuration will no longer be available for this Emitter and listening.
    //!
    //! \param[in] importerConfig  Configuration pointer.
    //----------------------------------------
    void unregisterImporterConfiguration(const ConfigurationPtr& importerConfig);

protected:
    //========================================
    //! \brief Add all conversions of the data package into data container by listener(s) of data type.
    //! \note Data type DataTypeId::DataType_Unknown is reserved for general data containers,
    //!       which are not mapped to a specific data type.
    //! \param[in] dataTypeId        DataTypeId for the listener filter.
    //! \param[in] dataPackage       Next data package.
    //! \param[in] dataContainers    All conversions of the next data package.
    //! \return Either \c true if conversions added to the vector, otherwise \c false.
    //----------------------------------------
    bool addDataContainers(const DataTypeId dataTypeId,
                           const IdcDataPackagePtr& dataPackage,
                           DataContainerImporterContextList& dataContainers);

private:
    //========================================
    //! \brief Notify data container listeners about imported data package.
    //! \param[in] dataContainerListenerMap     Access to data container listener map.
    //! \param[in] dataContainer                Data container which deserialized by importer.
    //! \param[in] importer                     Importer which used to deserialize data package.
    //! \param[in] configuration                Configuration which used for deserialization.
    //! \returns Either \c true if listeners are triggered or otherwise \c false.
    //----------------------------------------
    bool notifyDataContainerListeners(DataContainerListenerMapByDataType& dataContainerListenerMap,
                                      const DataContainerPtr& dataContainer,
                                      const ImporterPtr& importer,
                                      const ConfigurationPtr& configuration = nullptr);

private:
    //========================================
    //! \brief List of registered idc data package listeners.
    //----------------------------------------
    ThreadSafe<IdcDataPackageListenerList> m_IdcDataPackageListeners;

    //========================================
    //! \brief List of registered data container listeners.
    //----------------------------------------
    ThreadSafe<DataContainerListenerMapByDataType> m_dataContainerListeners;

    //========================================
    //! \brief List of registered importer configurations.
    //----------------------------------------
    ThreadSafe<ImporterConfigurationMapByType> m_importerConfigurations;
}; // class IdcEmitter

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
