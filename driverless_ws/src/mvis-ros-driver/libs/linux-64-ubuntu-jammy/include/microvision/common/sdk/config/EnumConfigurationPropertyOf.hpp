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
//! \date Mai 18, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/EnumConfigurationProperty.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements a enum configurable property of enum type \c ValueType.
//! \tparam ValueType  Enum value type.
//! \extends microvision::common::sdk::EnumConfigurationPropertyOf<ValueType>
//! \extends microvision::common::sdk::ConfigurationPropertyOf<ValueType>
//------------------------------------------------------------------------------
template<typename ValueType, typename = EnableIfEnum<ValueType>>
class EnumConfigurationPropertyOfEnumType : public virtual EnumConfigurationPropertyOf<ValueType>,
                                            public ConfigurationPropertyOf<ValueType>
{
public:
    //========================================
    //! \brief Base type definition.
    //----------------------------------------
    using BaseType = ConfigurationPropertyOf<ValueType>;

    //========================================
    //! \brief Enum base type definition.
    //----------------------------------------
    using EnumBaseType = EnumConfigurationPropertyOf<ValueType>;

    //========================================
    //! \brief Underlying type definition.
    //----------------------------------------
    using UnderlyingType = typename EnumBaseType::UnderlyingType;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~EnumConfigurationPropertyOfEnumType() override = default;

public: // getter
    //========================================
    //! \brief Get value of the property.
    //! \returns Value of property.
    //----------------------------------------
    Optional<ValueType> getValue() const override { return EnumBaseType::m_value.getValue(); }

    //========================================
    //! \brief Get default value of the property.
    //! \returns Default value of property.
    //----------------------------------------
    ValueType getDefaultValue() const override { return EnumBaseType::m_defaultValue; }

    //========================================
    //! \brief Get (default) value of the property.
    //! \returns (Default) value of property.
    //----------------------------------------
    ValueType getValueOrDefault() const override
    {
        auto valueAccess = EnumBaseType::m_value.get();

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
            auto valueAccess = EnumBaseType::m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                oldValue = **valueAccess;
            }

            if (value)
            {
                const auto it = this->m_valueNameMap.find(*value);

                if (it == this->m_valueNameMap.end())
                {
                    LOGWARNING(EnumConfigurationProperty::getLogger(),
                               "Enum value " << toHex(static_cast<UnderlyingType>(*value))
                                             << " is not mapped for configuration property " << this->getId());
                }
                else
                {
                    valueAccess = value;
                }
            }
            else
            {
                valueAccess = value;
            }
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
            auto valueAccess = EnumBaseType::m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                oldValue = **valueAccess;
            }

            if (value)
            {
                const auto it = this->m_valueNameMap.find(*value);

                if (it == this->m_valueNameMap.end())
                {
                    LOGWARNING(EnumConfigurationProperty::getLogger(),
                               "Enum value " << toHex(static_cast<UnderlyingType>(*value))
                                             << " is not mapped for configuration property " << this->getId());
                }
                else
                {
                    valueAccess = std::move(value);
                }
            }
            else
            {
                valueAccess = std::move(value);
            }
        }
        this->triggerOnSetValue(oldValue);
    }

}; // class EnumConfigurationPropertyOfEnumType

//==============================================================================
//! \brief Implements a enum underlying configurable property of enum type \c ValueType.
//! \tparam ValueType  Enum value type.
//! \extends microvision::common::sdk::EnumConfigurationPropertyOf<ValueType>
//! \extends microvision::common::sdk::ConfigurationPropertyOf<UnderlyingType>
//------------------------------------------------------------------------------
template<typename ValueType, typename = EnableIfEnum<ValueType>>
class EnumConfigurationPropertyOfUnderlyingType
  : public virtual EnumConfigurationPropertyOf<ValueType>,
    public ConfigurationPropertyOf<typename EnumConfigurationPropertyOf<ValueType>::UnderlyingType>
{
public:
    //========================================
    //! \brief Enum base type definition.
    //----------------------------------------
    using EnumBaseType = EnumConfigurationPropertyOf<ValueType>;

    //========================================
    //! \brief Underlying type definition.
    //----------------------------------------
    using UnderlyingType = typename EnumBaseType::UnderlyingType;

    //========================================
    //! \brief Base type definition.
    //----------------------------------------
    using BaseType = ConfigurationPropertyOf<UnderlyingType>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~EnumConfigurationPropertyOfUnderlyingType() override = default;

public: // getter
    //========================================
    //! \brief Get value of the property.
    //! \returns Value of property.
    //----------------------------------------
    Optional<UnderlyingType> getValue() const override
    {
        auto value = EnumBaseType::m_value.getValue();

        if (value)
        {
            return makeOptional<UnderlyingType>(static_cast<UnderlyingType>(*value));
        }
        else
        {
            return nullopt;
        }
    }

    //========================================
    //! \brief Get default value of the property.
    //! \returns Default value of property.
    //----------------------------------------
    UnderlyingType getDefaultValue() const override
    {
        return static_cast<UnderlyingType>(EnumBaseType::m_defaultValue);
    }

    //========================================
    //! \brief Get (default) value of the property.
    //! \returns (Default) value of property.
    //----------------------------------------
    UnderlyingType getValueOrDefault() const override
    {
        auto valueAccess = EnumBaseType::m_value.get();

        if (*valueAccess)
        {
            return static_cast<UnderlyingType>(**valueAccess);
        }
        else
        {
            return static_cast<UnderlyingType>(EnumBaseType::m_defaultValue);
        }
    }

public: // setter
    //========================================
    //! \brief Set value by copy.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(const UnderlyingType& value) override { this->setValue(Optional<UnderlyingType>{value}); }

    //========================================
    //! \brief Set value by move.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(UnderlyingType&& value) override { this->setValue(Optional<UnderlyingType>{std::move(value)}); }

    //========================================
    //! \brief Set value by copy value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(const Optional<UnderlyingType>& value) override
    {
        Any oldValue{};
        {
            auto valueAccess = EnumBaseType::m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                oldValue = static_cast<UnderlyingType>(**valueAccess);
            }

            if (value)
            {
                const auto it = this->m_valueNameMap.find(static_cast<ValueType>(*value));

                if (it == this->m_valueNameMap.end())
                {
                    LOGWARNING(EnumConfigurationProperty::getLogger(),
                               "Enum value " << toHex(*value) << " is not mapped for configuration property "
                                             << this->getId());
                }
                else
                {
                    valueAccess = Optional<ValueType>{static_cast<ValueType>(*value)};
                }
            }
            else
            {
                valueAccess = nullopt;
            }
        }
        this->triggerOnSetValue(oldValue);
    }

    //========================================
    //! \brief Set value by copy value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(Optional<UnderlyingType>&& value) override
    {
        Any oldValue{};
        {
            auto valueAccess = EnumBaseType::m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                oldValue = static_cast<UnderlyingType>(**valueAccess);
            }

            if (value)
            {
                const auto it = this->m_valueNameMap.find(static_cast<ValueType>(*value));

                if (it == this->m_valueNameMap.end())
                {
                    LOGWARNING(EnumConfigurationProperty::getLogger(),
                               "Enum value " << toHex(*value) << " is not mapped for configuration property "
                                             << this->getId());
                }
                else
                {
                    valueAccess = Optional<ValueType>{static_cast<ValueType>(*value)};
                }
            }
            else
            {
                valueAccess = nullopt;
            }
            value.reset();
        }
        this->triggerOnSetValue(oldValue);
    }

}; // class EnumConfigurationPropertyOfUnderlyingType

//==============================================================================
//! \brief Implements a enum name configurable property of enum type \c ValueType.
//! \tparam ValueType  Enum value type.
//! \extends microvision::common::sdk::EnumConfigurationPropertyOf<ValueType>
//! \extends microvision::common::sdk::ConfigurationPropertyOf<std::string>
//------------------------------------------------------------------------------
template<typename ValueType, typename = EnableIfEnum<ValueType>>
class EnumConfigurationPropertyOfEnumNameType : public virtual EnumConfigurationPropertyOf<ValueType>,
                                                public ConfigurationPropertyOf<std::string>
{
public:
    //========================================
    //! \brief Enum base type definition.
    //----------------------------------------
    using EnumBaseType = EnumConfigurationPropertyOf<ValueType>;

    //========================================
    //! \brief Base type definition.
    //----------------------------------------
    using BaseType = ConfigurationPropertyOf<std::string>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~EnumConfigurationPropertyOfEnumNameType() override = default;

public: // getter
    //========================================
    //! \brief Get value of the property.
    //! \returns Value of property.
    //----------------------------------------
    Optional<std::string> getValue() const override { return EnumBaseType::getNameByValue(); }

    //========================================
    //! \brief Get default value of the property.
    //! \returns Default value of property.
    //----------------------------------------
    std::string getDefaultValue() const override { return EnumBaseType::getNameByDefaultValue(); }

    //========================================
    //! \brief Get (default) value of the property.
    //! \returns (Default) value of property.
    //----------------------------------------
    std::string getValueOrDefault() const override { return EnumBaseType::getNameByValueOrDefault(); }

public: // setter
    //========================================
    //! \brief Set value by copy.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(const std::string& value) override { this->setValue(Optional<std::string>{value}); }

    //========================================
    //! \brief Set value by move.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(std::string&& value) override { this->setValue(Optional<std::string>{std::move(value)}); }

    //========================================
    //! \brief Set value by copy value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(const Optional<std::string>& value) override { EnumBaseType::setValueByName(value); }

    //========================================
    //! \brief Set value by copy value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    void setValue(Optional<std::string>&& value) override { EnumBaseType::setValueByName(std::move(value)); }

}; // class EnumConfigurationPropertyOfEnumNameType

//==============================================================================
//! \brief Combined all implementations of enum configurable property.
//! \note Select's the enum type implementation as primary.
//! \tparam ValueType  Enum value type.
//! \extends microvision::common::sdk::EnumConfigurationPropertyOfEnumType<ValueType>
//! \extends microvision::common::sdk::EnumConfigurationPropertyOfUnderlyingType<std::string>
//! \extends microvision::common::sdk::EnumConfigurationPropertyOfEnumNameType<std::string>
//------------------------------------------------------------------------------
template<typename ValueType, typename = EnableIfEnum<ValueType>>
class EnumConfigurationPropertyOfType final : public EnumConfigurationPropertyOfEnumType<ValueType>,
                                              public EnumConfigurationPropertyOfUnderlyingType<ValueType>,
                                              public EnumConfigurationPropertyOfEnumNameType<ValueType>
{
public:
    //========================================
    //! \brief Base type definition.
    //----------------------------------------
    using BaseType = EnumConfigurationPropertyOf<ValueType>;

    //========================================
    //! \brief Enum base type definition.
    //----------------------------------------
    using EnumBaseType = EnumConfigurationPropertyOfEnumType<ValueType>;

    //========================================
    //! \brief Initializer list of name value enum pairs.
    //----------------------------------------
    using ValueNameInitList = typename BaseType::ValueNameInitList;

public:
    //========================================
    //! \brief Construct configurable property with default value and enum name value map.
    //! \param[in] id               Unique id of the property.
    //! \param[in] label            Name of the property.
    //! \param[in] defaultValue     Default value.
    //! \param[in] valueNames       Enum name value map.
    //----------------------------------------
    EnumConfigurationPropertyOfType(const std::string id,
                                    const std::string label,
                                    const ValueType defaultValue,
                                    ValueNameInitList valueNameMap)
      : ConfigurationProperty(id, label), EnumConfigurationProperty(), BaseType(defaultValue, valueNameMap)
    {}

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    EnumConfigurationPropertyOfType(const EnumConfigurationPropertyOfType& other)
      : ConfigurationProperty(other), EnumConfigurationProperty(other), BaseType(other)
    {}

    //========================================
    //! \brief Move constructor (deleted).
    //----------------------------------------
    EnumConfigurationPropertyOfType(EnumConfigurationPropertyOfType&&) noexcept = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~EnumConfigurationPropertyOfType() override = default;

public: // implements ConfigurationProperty
    //========================================
    //! \brief Check if property value is set.
    //! \returns Either \c true if value is set or otherwise \c false.
    //----------------------------------------
    bool hasValue() const override { return static_cast<bool>(*BaseType::m_value.get()); }

    //========================================
    //! \brief Get property value type.
    //! \returns Value type info.
    //----------------------------------------
    const std::type_info& getType() const override { return typeid(EnumConfigurationProperty); }

public: // implements ConfigurationPropertyOf
    //========================================
    //! \brief Get value of the property.
    //----------------------------------------
    using EnumBaseType::getValue;

    //========================================
    //! \brief Get default value of the property.
    //----------------------------------------
    using EnumBaseType::getDefaultValue;

    //========================================
    //! \brief Get (default) value of the property.
    //----------------------------------------
    using EnumBaseType::getValueOrDefault;

    //========================================
    //! \brief Set value of optional.
    //----------------------------------------
    using EnumBaseType::setValue;

}; // class ConfigurationPropertyOfEnum<T>

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
