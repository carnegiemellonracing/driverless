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
//! \date Mai 14, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/ConfigurationProperty.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/misc/Optional.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Exception thrown if cast of configuration property reference failed.
//------------------------------------------------------------------------------
class BadConfigurationPropertyCast final : public std::bad_cast
{
public:
    //========================================
    //! \brief Constructs exception and message with source and target type arguments.
    //! \param[in] fromType  Source type of any value.
    //! \param[in] toType    Target type of cast.
    //----------------------------------------
    BadConfigurationPropertyCast(const std::type_info& fromType, const std::type_info& toType)
    {
        this->m_message.append("Cannot cast value of type '")
            .append(fromType.name())
            .append("' to type '")
            .append(toType.name())
            .append("'.");
    }

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BadConfigurationPropertyCast() override = default;

public:
    //========================================
    //! \brief Get's error message.
    //! \returns Error message.
    //----------------------------------------
    const char* what() const noexcept override { return this->m_message.c_str(); }

private:
    //========================================
    //! \brief Error message.
    //----------------------------------------
    std::string m_message{};
}; // class BadConfigurationPropertyCast

//==============================================================================
//! \brief Abstract configurable property base of type ValueType.
//! \tparam ValueType  Any value type.
//! \extends microvision::common::sdk::ConfigurationProperty
//------------------------------------------------------------------------------
template<typename ValueType>
class ConfigurationPropertyOf : public virtual ConfigurationProperty
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ConfigurationPropertyOf() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~ConfigurationPropertyOf() = default;

public: // getter
    //========================================
    //! \brief Get value of the property.
    //! \returns Value of property.
    //----------------------------------------
    virtual Optional<ValueType> getValue() const = 0;

    //========================================
    //! \brief Get default value of the property.
    //! \returns Default value of property.
    //----------------------------------------
    virtual ValueType getDefaultValue() const = 0;

    //========================================
    //! \brief Get (default) value of the property.
    //! \returns (Default) value of property.
    //----------------------------------------
    virtual ValueType getValueOrDefault() const = 0;

public: // setter
    //========================================
    //! \brief Set value by copy.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    virtual void setValue(const ValueType& value) = 0;

    //========================================
    //! \brief Set value by move.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    virtual void setValue(ValueType&& value) = 0;

    //========================================
    //! \brief Set value by copy value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    virtual void setValue(const Optional<ValueType>& value) = 0;

    //========================================
    //! \brief Set value by move value of optional.
    //! \param[in] value  New value of property type.
    //----------------------------------------
    virtual void setValue(Optional<ValueType>&& value) = 0;

}; // class ConfigurationPropertyOf<T>

//==============================================================================
//! \brief Cast property pointer into ConfigurationPropertyOf<ValueType> pointer.
//! \tparam ValueType  Any value type.
//! \param[in] property  Property pointer to cast into ValueType.
//! \returns Either \c ConfigurationPropertyOf<ValueType> pointer if castable or otherwise \c nullptr.
//------------------------------------------------------------------------------
template<typename ValueType>
inline ConfigurationPropertyOf<ValueType>* castProperty(ConfigurationProperty* property)
{
    return dynamic_cast<ConfigurationPropertyOf<ValueType>*>(property);
}

//==============================================================================
//! \brief Cast property pointer into ConfigurationPropertyOf<ValueType> pointer.
//! \tparam ValueType  Any value type.
//! \param[in] property  Property pointer to cast into ValueType.
//! \returns Either \c ConfigurationPropertyOf<ValueType> pointer if castable or otherwise \c nullptr.
//------------------------------------------------------------------------------
template<typename ValueType>
inline const ConfigurationPropertyOf<ValueType>* castProperty(const ConfigurationProperty* property)
{
    return dynamic_cast<const ConfigurationPropertyOf<ValueType>*>(property);
}

//==============================================================================
//! \brief Cast property reference into ConfigurationPropertyOf<ValueType> reference.
//! \tparam ValueType  Any value type.
//! \param[in] property  Property reference to cast into ValueType.
//! \throws BadConfigurationPropertyCast  Will be thrown if cast failed.
//! \returns \c ConfigurationPropertyOf<ValueType> reference.
//------------------------------------------------------------------------------
template<typename ValueType>
inline ConfigurationPropertyOf<ValueType>& castProperty(ConfigurationProperty& property)
{
    auto* castedProperty = castProperty<ValueType>(&property);

    if (castedProperty == nullptr)
    {
        throw BadConfigurationPropertyCast(property.getType(), typeid(ValueType));
    }

    return *castedProperty;
}

//==============================================================================
//! \brief Cast property reference into ConfigurationPropertyOf<ValueType> reference.
//! \tparam ValueType  Any value type.
//! \param[in] property  Property reference to cast into ValueType.
//! \throws BadConfigurationPropertyCast  Will be thrown if cast failed.
//! \returns \c ConfigurationPropertyOf<ValueType> reference.
//------------------------------------------------------------------------------
template<typename ValueType>
inline const ConfigurationPropertyOf<ValueType>& castProperty(const ConfigurationProperty& property)
{
    const auto* castedProperty = castProperty<ValueType>(&property);

    if (castedProperty == nullptr)
    {
        throw BadConfigurationPropertyCast(property.getType(), typeid(ValueType));
    }

    return *castedProperty;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
