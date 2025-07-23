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
//! \date Jan 10, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/ImporterBase.hpp>

#include <microvision/common/sdk/DataTypeId.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//!\class DataContainerListenerBase
//!\brief Abstract base class for all listener.
//!
//! A DataContainerListener can be registered to an IdcDevice to receive
//! all DataContainers of the associated DataType received by that
//! device.
//!
//! Derived classes will have to implement an onData method for the
//! associated DataType. This onData method will be called in the
//! context of the receive thread of that device.
//!
//! The data received by the onData method will be deserialized.
//!
//! In case one is not interested in the contents of that DataContainer
//! it may be better to implement a DataStreamer.
//!
//!\sa DataStreamer
//------------------------------------------------------------------------------
class DataContainerListenerBase
{
public:
    //========================================
    //! \brief Destrutor does nothing special.
    //----------------------------------------
    virtual ~DataContainerListenerBase() {}

public:
    using HashId           = uint64_t;
    using DataTypeHashUuid = DataContainerBase::IdentificationKey;
    using DataTypes        = std::vector<DataTypeHashUuid>;

    //========================================
    //! \brief Get the DataTypes for which this
    //!        listener is listening.
    //! \return The DataTypes the listener is
    //!         listening for.
    //----------------------------------------
    const DataTypes& getAssociatedDataTypes() const { return m_registeredDataTypes; }

protected:
    //========================================
    //! \brief Register a data type for listening.
    //!
    //! The listener keeps track of all the types registered for listening.
    //!
    //! \param[in] type  idc data type part of the registered type.
    //! \param[in] hash  Hash of class identification string part of the registered type.
    //! \param[in] uuid  Custom data container Uuid part of the registered type.
    //----------------------------------------
    void registerDataType(const DataTypeId type, const HashId hash, DataContainerBase::Uuid uuid)
    {
        // register listener for the given type if not already registered
        const auto checkRegistered = [type, hash, uuid](const DataTypeHashUuid& entry) {
            return (entry.getId() == type) && (entry.getHash() == hash) && (entry.getUuid() == uuid);
        };
        const auto foundIt = std::find_if(m_registeredDataTypes.begin(), m_registeredDataTypes.end(), checkRegistered);
        if (foundIt == m_registeredDataTypes.end())
        {
            m_registeredDataTypes.push_back(DataTypeHashUuid(type, hash, uuid));
        }
    }

    //========================================
    //! \brief Register a data type for listening.
    //!
    //! This overload can only be used to register standard general or special types.
    //! For custom data containers this cannot be used because of the missing uuid.
    //!
    //! The listener keeps track of all the types registered for listening.
    //!
    //! \param[in] type  idc data type part of the registered type.
    //! \param[in] hash  Hash of class identification string part of the registered type.
    //----------------------------------------
    void registerDataType(const DataTypeId type, const HashId hash)
    {
        // register standard idc data container types
        registerDataType(type, hash, DataContainerBase::Uuid());
    }

    //========================================
    //! \brief Register a data type for listening.
    //!
    //! This overload can only be used to register standard general or special types.
    //! For custom data containers this cannot be used because of the missing uuid.
    //!
    //! The listener keeps track of all the types registered for listening.
    //!
    //! \param[in] hash  Hash of class identification string part of the registered type.
    //----------------------------------------
    void registerDataType(const HashId hash)
    {
        // Register the container with the generic data type.
        registerDataType(DataTypeId::DataType_Unknown, hash);
    }

    //========================================
    //! \brief Register a custom data container type for listening.
    //!
    //! This overload can only be used to register custom data container types.
    //!
    //! The listener keeps track of all the types registered for listening.
    //!
    //! \param[in] uuid  Custom data container Uuid part of the registered type.
    //----------------------------------------
    void registerDataType(DataContainerBase::Uuid uuid)
    {
        // register some custom data type identified by uuid
        registerDataType(
            DataTypeId::DataType_CustomDataContainer, CustomDataContainerBase::getClassIdHashStatic(), uuid);
    }

private:
    DataTypes m_registeredDataTypes; // what data types this is listing for
}; // DataContainerListenerBase

//==============================================================================

//==============================================================================
//!\class DataContainerListener
//!\brief Abstract base class for all DataListener listen on DataContainerImpl.
//!
//!\tparam DataContainerImpl  DataContainer implementation.
//!\tparam dataType           Data type.
//------------------------------------------------------------------------------
template<class DataContainerImpl, uint16_t dataType>
class DataContainerListener : public virtual DataContainerListenerBase
{
public:
    //========================================
    //!\brief Constructor registers at DataListenerBase class
    //----------------------------------------
    DataContainerListener()
    {
        // custom data container -> register via uuid
        registerDataType(
            DataTypeId(dataType), DataContainerImpl::getClassIdHashStatic(), findUuid<DataContainerImpl>());
    }

    //========================================
    //! \brief Called on receiving a new DataContainerImpl data container.
    //!
    //! Method to be called by the IdcDevice where this listener
    //! is registered when a new DataContainer of type DataContainerImpl
    //! has been received.
    //!
    //! \param[in] dataContainerPtr     Shared pointer to an instance of DataContainerImpl that has been received.
    //! \param[in] configuration        (Optional) Configuration context for import. Default set as nullptr.
    //----------------------------------------
    virtual void onData(std::shared_ptr<const DataContainerImpl> dataContainerPtr,
                        const ConfigurationPtr& configuration)
        = 0;

}; // DataContainerListener

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================

//==============================================================================
//!\class GeneralDataContainerListener
//!\brief Abstract base class for all DataContainerListener listen on DataContainerImpl.
//!
//!\tparam DataContainerImpl  DataContainer implementation.
//------------------------------------------------------------------------------
template<class DataContainerImpl>
class GeneralDataContainerListener : public virtual DataContainerListenerBase
{
public:
    //========================================
    //!\brief Constructor registers at DataListenerBase class
    //----------------------------------------
    GeneralDataContainerListener() { registerDataType(DataContainerImpl::getClassIdHashStatic()); }

    //========================================
    //!\brief Called on receiving a new DataContainerImpl DataContainer.
    //!
    //!Method to be called by the IdcDevice where this listener
    //!is registered when a new DataContainer of type DataContainerImpl
    //!has been received.
    //!
    //! \param[in] dataContainerPtr     Pointer to the DataContainerImpl that has been received.
    //! \param[in] configuration        (Optional) Configuration context for import. Default set as nullptr.
    //----------------------------------------
    virtual void onData(std::shared_ptr<const DataContainerImpl> dataContainerPtr,
                        const ConfigurationPtr& configuration)
        = 0;

}; // GeneralDataContainerListener

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//!\brief Abstract template set of base classes for all special listener.
//!\tparam DataContainerImpl  The DataContainer that will be read in case the
//!                           datatype has been received which contains the
//!                           special sub datatype which can be distinguished
//!                           using the value of \a KeyType.
//!\tparam KeyTpye            Type of the key variable used to distinguish between
//!                           between different special sub data types.
//------------------------------------------------------------------------------
template<class DataContainerImpl, typename KeyType>
class SpecialListenerBase
{
public:
    using HashId    = uint64_t;
    using KeyVector = std::vector<std::pair<KeyType, HashId>>;
    using KeyMap    = std::unordered_map<KeyType, KeyVector>;

    //========================================
    //!\brief Get the DataTypes for which this
    //!       listener is listening.
    //!\return The DataTypes the listener is
    //!        listening for.
    //----------------------------------------
    const KeyMap& getRegisteredSubTypes() const { return m_registeredSubTypes; }

protected:
    void registerKey(const KeyType subId, const HashId hash)
    {
        const auto newEntry = std::make_pair(subId, hash);
        std::cerr << "Register: " << subId << std::endl;

        KeyVector& kv = m_registeredSubTypes[subId];

        if (std::find(kv.begin(), kv.end(), newEntry) == kv.end())
        {
            std::cerr << "Register added: (" << newEntry.first << ", " << newEntry.second << ")" << std::endl;
            kv.push_back(newEntry);
        }

        std::cerr << "m_registeredSubTypes: ";
        for (auto r : kv)
        {
            std::cerr << "(" << r.first << ", " << r.second << ")  ";
        }
        std::cerr << std::endl;
    }

private:
    KeyMap m_registeredSubTypes;
}; // SpecialListenerBase

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//!\class DataContainerSpecialListener
//!\brief Abstract base class for all DataContainerListener listen on DataContainerImpl.
//!\date Feb 10, 2014
//!
//!\tparam DataContainerImpl  DataContainer implementation.
//!\tparam dataType           Data type.
//!\tparam SpecialType        Sub-type in data container.
//------------------------------------------------------------------------------
template<class DataContainerImpl, uint16_t dataType, class SpecialType>
class DataContainerSpecialListener
  : public virtual DataContainerListenerBase,
    public virtual SpecialListenerBase<DataContainerImpl, typename SpecialType::KeyType>
{
public:
    //========================================
    //!\brief Constructor registers at DataContainerListenerBase class
    //----------------------------------------
    DataContainerSpecialListener() : SpecialListenerBase<DataContainerImpl, typename SpecialType::KeyType>()
    {
        registerDataType(DataTypeId(dataType), DataContainerImpl::getClassIdHashStatic());
        this->registerKey(SpecialType::key, SpecialType::getClassIdHashStatic()); // gcc needs this pointer here.
    }

    //========================================
    //!\brief Called on receiving a new DataContainerImpl DataContainer.
    //!
    //!Method to be called by the IdcDevice where this listener
    //!is registered when a new DataContainer of type DataContainerImpl
    //!has been received.
    //!
    //!\param[in] dbImpl  Pointer to the DataContainerImpl that has
    //!                   been received.
    //----------------------------------------
    virtual void onData(const SpecialType* const stImpl) = 0;
}; // DataContainerSpecialListener

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
