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
//! \date Jan 27, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/ConfigurationPropertyOf.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/misc/Optional.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Defines a configurable property of type ValueType.
//! \tparam ValueType  Any value type.
//! \extends microvision::common::sdk::ConfigurationPropertyOf<ValueType>
//------------------------------------------------------------------------------
template<typename ValueType>
class ConfigurationPropertyOfType final : public ConfigurationPropertyOf<ValueType>
{
public:
    //========================================
    //! \brief Base type definition.
    //----------------------------------------
    using BaseType = ConfigurationPropertyOf<ValueType>;

public:
    //========================================
    //! \brief Construct configurable property with default value.
    //! \param[in] id               Unique id of the property.
    //! \param[in] label            Name of the property.
    //! \param[in] defaultValue     Default value.
    //----------------------------------------
    ConfigurationPropertyOfType(const std::string id, const std::string label, const ValueType defaultValue)
      : ConfigurationProperty(id, label), BaseType(), m_value{}, m_defaultValue{defaultValue}
    {}

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    ConfigurationPropertyOfType(const ConfigurationPropertyOfType& other)
      : ConfigurationProperty{other},
        BaseType{other},
        m_value{other.m_value.getValue()},
        m_defaultValue{other.m_defaultValue}
    {}

    //========================================
    //! \brief Move constructor (deleted).
    //----------------------------------------
    ConfigurationPropertyOfType(ConfigurationPropertyOfType&& other) noexcept = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ConfigurationPropertyOfType() override = default;

public: // implements ConfigurationProperty
    //========================================
    //! \brief Get property value type.
    //! \returns Value type info.
    //----------------------------------------
    const std::type_info& getType() const override { return typeid(ValueType); }

    //========================================
    //! \brief Check if property value is set.
    //! \returns Either \c true if value is set or otherwise \c false.
    //----------------------------------------
    bool hasValue() const override { return static_cast<bool>(*this->m_value.get()); }

    //========================================
    //! \brief Copy configuration property value from another configuration property.
    //! \param[in] other  Other configuration property to copy value from.
    //! \return Either \c true if value could been copied, otherwise \c false if not.
    //----------------------------------------
    bool copyValueFrom(const ConfigurationProperty& other) override
    {
        const auto property = castProperty<ValueType>(&other);

        if (property != nullptr)
        {
            this->setValue(property->getValue());
            return true;
        }
        return false;
    }

public: // getter
    //========================================
    //! \brief Get value of the property.
    //! \returns Value of property.
    //----------------------------------------
    Optional<ValueType> getValue() const override { return this->m_value.getValue(); }

    //========================================
    //! \brief Get default value of the property.
    //! \returns Default value of property.
    //----------------------------------------
    ValueType getDefaultValue() const override { return this->m_defaultValue; }

    //========================================
    //! \brief Get (default) value of the property.
    //! \returns (Default) value of property.
    //----------------------------------------
    ValueType getValueOrDefault() const override
    {
        auto valueAccess = this->m_value.get();

        if (*valueAccess)
        {
            return **valueAccess;
        }
        else
        {
            return this->m_defaultValue;
        }
    }

public: // setter
    //========================================
    //! \brief Set value by copy.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(const ValueType& value) override { this->setValue(Optional<ValueType>{value}); }

    //========================================
    //! \brief Set value by move.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(ValueType&& value) override { this->setValue(Optional<ValueType>{std::move(value)}); }

    //========================================
    //! \brief Set value by copy value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(const Optional<ValueType>& value) override
    {
        Any oldValue{};
        {
            auto valueAccess = this->m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                oldValue = **valueAccess;
            }

            valueAccess = value;
        }
        this->triggerOnSetValue(oldValue);
    }

    //========================================
    //! \brief Set value by move value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(Optional<ValueType>&& value) override
    {
        Any oldValue{};
        {
            auto valueAccess = this->m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                oldValue = **valueAccess;
            }

            valueAccess = std::move(value);
        }
        this->triggerOnSetValue(oldValue);
    }

private:
    //========================================
    //! \brief Value of property.
    //----------------------------------------
    ThreadSafe<Optional<ValueType>> m_value;

    //========================================
    //! \brief Default value of property.
    //----------------------------------------
    const ValueType m_defaultValue;

}; // class ConfigurationPropertyOfType<T>

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
