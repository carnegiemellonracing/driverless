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

#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/misc/Any.hpp>

#include <functional>
#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Abstract base class of configurable property to provide type unspecific interface.
//!
//! A configuration uses a map of these to hold configuration values of different types. Each one has a unique id.
//! \sa microvision::common::sdk::Configuration
//------------------------------------------------------------------------------
class ConfigurationProperty
{
public:
    //========================================
    //! \brief Function definition for the on set value event listener.
    //!
    //! As parameter will provide the changed config entry and the previous value.
    //----------------------------------------
    using OnSetValueEventListenerFunction = void (*)(const ConfigurationProperty&, const Any&);

    //========================================
    //! \brief Function definition for the on set value event listener.
    //!
    //! As parameter will provide the changed config entry and the previous value.
    //----------------------------------------
    using OnSetValueEventListener = std::function<void(const ConfigurationProperty&, const Any&)>;

protected:
    //========================================
    //! \brief Empty constructor to fix virtual inheritance tree instantiation.
    //! \notes Should never be used in runtime, is just to fix compile checks.
    //----------------------------------------
    ConfigurationProperty();

public:
    //========================================
    //! \brief Construct abstract base class of configurable property.
    //! \param[in] id               Unique id of the property.
    //! \param[in] label            Name of the property.
    //----------------------------------------
    ConfigurationProperty(const std::string id, const std::string label);

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    ConfigurationProperty(const ConfigurationProperty& entry);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    ConfigurationProperty(ConfigurationProperty&& entry) noexcept = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~ConfigurationProperty() = default;

public: // getter
    //========================================
    //! \brief Get unique id of the property.
    //! \return Unique id string.
    //----------------------------------------
    const std::string& getId() const;

    //========================================
    //! \brief Get name of the property.
    //! \return Name string.
    //----------------------------------------
    const std::string& getLabel() const;

    //========================================
    //! \brief Get property value type.
    //! \return Value type info.
    //----------------------------------------
    virtual const std::type_info& getType() const = 0;

    //========================================
    //! \brief Check if property value is set.
    //! \return Either \c true if value is set or otherwise \c false.
    //----------------------------------------
    virtual bool hasValue() const = 0;

public:
    //========================================
    //! \brief Copy configuration property value from another configuration property.
    //! \param[in] other  Other configuration property to copy value from.
    //! \return Either \c true if value could been copied, otherwise \c false if not.
    //----------------------------------------
    virtual bool copyValueFrom(const ConfigurationProperty& other) = 0;

public:
    //========================================
    //! \brief Register event listener for the on set value event.
    //!
    //! The on set value event will trigger if the value has set.
    //!
    //! \param[in] listener  New listener which will call if value has changed.
    //----------------------------------------
    void registerOnSetValueEventListener(const OnSetValueEventListener& listener);

    //========================================
    //! \brief Unregister event listener from the on set value event.
    //!
    //! The on set value event will trigger if the value has set.
    //!
    //! \param[in] listener  Old listener which will not call if value has set.
    //----------------------------------------
    void unregisterOnSetValueEventListener(const OnSetValueEventListener& listener);

protected:
    //========================================
    //! \brief Check if onSetValueEventListeners are registered.
    //! \return Either \c true if listeners are registered or otherwise \c false.
    //----------------------------------------
    bool hasOnSetValueEventListener() const;

    //========================================
    //! \brief Call all listener that the value has set.
    //!
    //! The on set value event will trigger if the value has set.
    //!
    //! \param[in] oldValue  Old value before set.
    //----------------------------------------
    void triggerOnSetValue(const Any& oldValue);

private:
    //========================================
    //! \brief Unique if of property.
    //----------------------------------------
    const std::string m_id;

    //========================================
    //! \brief Name of property.
    //----------------------------------------
    const std::string m_label;

    //========================================
    //! \brief All listeners they will call if value has set.
    //----------------------------------------
    ThreadSafe<std::list<OnSetValueEventListener>> m_onSetValueEventListeners;
}; // class ConfigurationProperty

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
