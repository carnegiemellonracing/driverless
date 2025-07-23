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
//! \date Nov 5, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/ConfigurationPropertyOf.hpp>

#include <microvision/common/sdk/misc/SharedBuffer.hpp>

#include <microvision/common/sdk/io.hpp>

#include <microvision/common/logging/logging.hpp>

#include <unordered_map>

#include <typeindex>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Forward declaration of Configuration for ConfigurationPtr using.
//------------------------------------------------------------------------------
class Configuration;

//==============================================================================
//! \brief Nullable shared Configuration pointer.
//------------------------------------------------------------------------------
using ConfigurationPtr = std::shared_ptr<Configuration>;

//==============================================================================
//! \brief Nullable unique Configuration pointer
//------------------------------------------------------------------------------
using ConfigurationUPtr = std::unique_ptr<Configuration>;

//==============================================================================
//! \brief Abstract base class which provides the interface for configuration implementations.
//! \sa microvision::common::sdk::Configurable
//------------------------------------------------------------------------------
class Configuration
{
private:
    //========================================
    //! \brief Type of storage for all ConfigurationProperty pointers indexed by unique string id.
    //----------------------------------------
    using StoreType = std::unordered_map<std::string, ConfigurationProperty*>;

public:
    //========================================
    //! \brief ConfigurationProperty iterator.
    //!
    //! Wraps the StoredType::iterator.
    //----------------------------------------
    class Iterator
    {
        friend class Configuration;

    private:
        //========================================
        //! \brief Construct iterator from StoredType::iterator.
        //! \param[in] it  Value of StoredType::iterator.
        //----------------------------------------
        Iterator(StoreType::iterator it) : m_iterator{it} {}

    public:
        //========================================
        //! \brief Equal compare of two iterators.
        //! \param[in] lhs  Iterator of set.
        //! \param[in] rhs  Iterator of set.
        //! \return Either \c true if the iterator is at the same position in set
        //!                 or otherwise \c false.
        //----------------------------------------
        friend bool operator==(const Iterator& lhs, const Iterator& rhs) { return lhs.m_iterator == rhs.m_iterator; }

        //========================================
        //! \brief Unequal compare of two iterators.
        //! \param[in] lhs  Iterator of set.
        //! \param[in] rhs  Iterator of set.
        //! \return Either \c true if the iterator is not at the same position in set
        //!                 or otherwise \c false.
        //----------------------------------------
        friend bool operator!=(const Iterator& lhs, const Iterator& rhs) { return lhs.m_iterator != rhs.m_iterator; }

        //========================================
        //! \brief Pre increment the iterator by one.
        //! \return Reference of incremented iterator.
        //----------------------------------------
        Iterator& operator++()
        {
            ++this->m_iterator;
            return *this;
        }

        //========================================
        //! \brief Post increment the iterator by one.
        //! \return Copy of iterator before incremented.
        //----------------------------------------
        Iterator operator++(int)
        {
            Iterator it = *this;
            ++this->m_iterator;
            return it;
        }

        //========================================
        //! \brief Get ConfigurationProperty from position.
        //! \return Pointer to ConfigurationProperty.
        //----------------------------------------
        ConfigurationProperty* operator->() { return this->m_iterator->second; }

        //========================================
        //! \brief Get ConfigurationProperty from position.
        //! \return Reference to ConfigurationProperty.
        //----------------------------------------
        ConfigurationProperty& operator*() { return *this->m_iterator->second; }

        //========================================
        //! \brief Get ConfigurationProperty from position.
        //! \return Pointer to ConfigurationProperty.
        //----------------------------------------
        ConfigurationProperty* get() { return this->m_iterator->second; }

    private:
        //========================================
        //! \brief Wrapped store iterator.
        //----------------------------------------
        StoreType::iterator m_iterator;
    };

    //========================================
    //! \brief Readonly ConfigurationProperty iterator.
    //!
    //! Wraps the StoredType::const_iterator.
    //----------------------------------------
    class ConstIterator
    {
        friend class Configuration;

    private:
        //========================================
        //! \brief Construct iterator from StoredType::const_iterator.
        //! \param[in] it  Value of StoredType::const_iterator.
        //----------------------------------------
        ConstIterator(StoreType::const_iterator it) : m_constIterator{it} {}

    public:
        //========================================
        //! \brief Equal compare of two iterators.
        //! \param[in] lhs  ConstIterator of set.
        //! \param[in] rhs  ConstIterator of set.
        //! \return Either \c true if the iterator is at the same position in set
        //!                 or otherwise \c false.
        //----------------------------------------
        friend bool operator==(const ConstIterator& lhs, const ConstIterator& rhs)
        {
            return lhs.m_constIterator == rhs.m_constIterator;
        }

        //========================================
        //! \brief Unequal compare of two iterators.
        //! \param[in] lhs  ConstIterator of set.
        //! \param[in] rhs  ConstIterator of set.
        //! \return Either \c true if the iterator is not at the same position in set
        //!                 or otherwise \c false.
        //----------------------------------------
        friend bool operator!=(const ConstIterator& lhs, const ConstIterator& rhs)
        {
            return lhs.m_constIterator != rhs.m_constIterator;
        }

        //========================================
        //! \brief Pre increment the iterator by one.
        //! \return Reference of incremented iterator.
        //----------------------------------------
        const ConstIterator& operator++()
        {
            ++this->m_constIterator;
            return *this;
        }

        //========================================
        //! \brief Post increment the iterator by one.
        //! \return Copy of iterator before increment.
        //----------------------------------------
        ConstIterator operator++(int)
        {
            ConstIterator it = *this;
            ++this->m_constIterator;
            return it;
        }

        //========================================
        //! \brief Get ConfigurationProperty from position.
        //! \return Pointer to ConfigurationProperty.
        //----------------------------------------
        const ConfigurationProperty* operator->() const { return this->m_constIterator->second; }

        //========================================
        //! \brief Get ConfigurationProperty from position.
        //! \return Reference to ConfigurationProperty.
        //----------------------------------------
        const ConfigurationProperty& operator*() const { return *this->m_constIterator->second; }

        //========================================
        //! \brief Get ConfigurationProperty at current position.
        //! \return Pointer to ConfigurationProperty.
        //----------------------------------------
        const ConfigurationProperty* get() const { return this->m_constIterator->second; }

    private:
        //========================================
        //! \brief Wrapped store iterator.
        //----------------------------------------
        StoreType::const_iterator m_constIterator;
    };

protected:
    //========================================
    //! \brief Logger
    //----------------------------------------
    static logging::LoggerSPtr logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    Configuration();

    //========================================
    //! \brief Copy constructor which does not copy property links.
    //----------------------------------------
    Configuration(const Configuration&);

    //========================================
    //! \brief Disabled move constructor to ensure thread-safety.
    //----------------------------------------
    Configuration(Configuration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~Configuration() = default;

public:
    //========================================
    //! \brief Get type of configuration to match with.
    //! \return Configuration type.
    //----------------------------------------
    virtual const std::string& getType() const = 0;

    //========================================
    //! \brief Get copy of configuration.
    //! \note Use copy method to avoid unreachable copy constructor which by unknown implementation type.
    //! \return Pointer to new copied Configuration.
    //----------------------------------------
    virtual ConfigurationPtr copy() const = 0;

    //========================================
    //! \brief Get memory size in serialized state.
    //! \return Serialized size.
    //! \note A return value of 0 means that the configuration is not supposed to be serialized.
    //----------------------------------------
    virtual std::size_t getSerializedSize() const;

    //========================================
    //! \brief Serialize confiuration into stream.
    //! \param[out]  os  Output stream.
    //! \returns Whether serialization was a success.
    //--------------------------------------
    virtual bool serialize(std::ostream& os) const;

    //========================================
    //! \brief Deserialize from stream.
    //! \param[out]  is  Input stream.
    //! \returns Whether deserialization was a success.
    //--------------------------------------
    virtual bool deserialize(std::istream& is);

public:
    //========================================
    //! \brief Get an iterator of position where find the ConfigurationProperty by unique id.
    //! \param[in] id  Unique id of the ConfigurationProperty.
    //! \return Iterator of the position where the ConfigurationProperty is or if not found the end Iterator.
    //----------------------------------------
    Iterator find(const std::string& id);

    //========================================
    //! \brief Get a readonly iterator of position where find the ConfigurationProperty by unique id.
    //! \param[in] id  Unique id of the ConfigurationProperty.
    //! \return ConstIterator of the position where the ConfigurationProperty is or if not found the end ConstIterator.
    //----------------------------------------
    const ConstIterator cfind(const std::string& id) const;

    //========================================
    //! \brief Get an iterator of the first position in set.
    //! \return Iterator of the first position in set or if empty the end Iterator.
    //----------------------------------------
    Iterator begin();

    //========================================
    //! \brief Get an iterator of the first position in set.
    //! \return ConstIterator of the first position in set or if empty the end ConstIterator.
    //----------------------------------------
    const ConstIterator cbegin() const;

    //========================================
    //! \brief Get an iterator of the end position of the set.
    //! \note These iterator does not contain an ConfigurationProperty.
    //! \return Iterator of the end position of the set.
    //----------------------------------------
    Iterator end();

    //========================================
    //! \brief Get an iterator of the end position of the set.
    //! \note These iterator does not contain an ConfigurationProperty.
    //! \return ConstIterator of the end position of the set.
    //----------------------------------------
    const ConstIterator cend() const;

    //========================================
    //! \brief Get the size of the set.
    //! \return Size of the set.
    //----------------------------------------
    std::size_t size() const noexcept;

public:
    //========================================
    //! \brief Copy configuration property values from another configuration.
    //! \param[in] other  Other configuration to copy values from.
    //! \return Number of property values which could been copied.
    //----------------------------------------
    std::size_t copyValuesFrom(const Configuration& other);

public:
    //========================================
    //! \brief Try to get configuration property value or default value by id.
    //!
    //! \tparam ValueType  Type of configuration property value.
    //! \param[in]   id         Configuration property id.
    //! \param[out]  outValue   Instance to assign value or default value of configuration property.
    //! \return Either \c true if configuration property found an value is of type \a ValueType or otherwise \c false.
    //!
    //! \note If no configuration property with that id exists it will not assign the outValue.
    //----------------------------------------
    template<typename ValueType>
    bool tryGetValueOrDefault(const std::string& id, ValueType& outValue) const
    {
        const auto propertyIterator = this->cfind(id);

        if (propertyIterator != this->cend())
        {
            const auto* property = castProperty<ValueType>(propertyIterator.get());

            if (property != nullptr)
            {
                outValue = property->getValueOrDefault();
                return true;
            }
        }

        return false;
    }

    //========================================
    //! \brief Try to set configuration property value by id.
    //!
    //! \tparam ValueType   Type of configuration property value.
    //! \param[in]  id          Configuration property id.
    //! \param[in]  inValue     New value of configuration property.
    //! \return Either \c true if configuation property found an value is of type \a ValueType or otherwise \c false.
    //!
    //! \note If no configuration property with that id exists it will not set the inValue.
    //----------------------------------------
    template<typename ValueType>
    bool trySetValue(const std::string& id, const ValueType& inValue)
    {
        auto propertyIterator = this->find(id);

        if (propertyIterator != this->end())
        {
            auto* property = castProperty<ValueType>(propertyIterator.get());

            if (property != nullptr)
            {
                property->setValue(inValue);
                return true;
            }
        }

        return false;
    }

    //========================================
    //! \brief Try to unset configuration property value by id.
    //!
    //! \tparam ValueType   Type of configuration property value.
    //! \param[in]  id          Configuration property id.
    //! \return Either \c true if configuation property found an value is of type \a ValueType or otherwise \c false.
    //!
    //! \note If no configuration property with that id exists it will not unset the value.
    //----------------------------------------
    template<typename ValueType>
    bool tryUnsetValue(const std::string& id)
    {
        auto propertyIterator = this->find(id);

        if (propertyIterator != this->end())
        {
            auto* property = castProperty<ValueType>(propertyIterator.get());

            if (property != nullptr)
            {
                property->setValue(Optional<ValueType>{nullopt});
                return true;
            }
        }

        return false;
    }

public:
    //========================================
    //! \brief Register event listener for all on set value events.
    //!
    //! The on set value event will trigger if the value has set.
    //!
    //! \param[in] listener  New listener which will call if value has changed.
    //----------------------------------------
    void registerOnSetValueEventListener(const ConfigurationProperty::OnSetValueEventListener& listener);

    //========================================
    //! \brief Unregister event listener from all on set value events.
    //!
    //! The on set value event will trigger if the value has set.
    //!
    //! \param[in] listener  Old listener which will not call if value has set.
    //----------------------------------------
    void unregisterOnSetValueEventListener(const ConfigurationProperty::OnSetValueEventListener& listener);

protected:
    //========================================
    //! \brief Update or add property to register properties in derived classes.
    //! \param[in, out] property  Property to set on properties map.
    //----------------------------------------
    void updateProperty(ConfigurationProperty& property);

private:
    //========================================
    //! \brief Properties map, indexed by property id.
    //----------------------------------------
    StoreType m_properties;

}; // class Configuration

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
