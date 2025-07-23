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

#include <microvision/common/sdk/config/ConfigurationPropertyOf.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/misc/Optional.hpp>
#include <microvision/common/sdk/misc/ToHex.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Enable it, if \c ValueType is of enum type.
//! \tparam ValueType  Enum value type.
//------------------------------------------------------------------------------
template<typename ValueType>
using EnableIfEnum = typename std::enable_if<std::is_enum<ValueType>::value>::type;

//==============================================================================
//! \brief Abstract base of enum configurable property.
//! \extends microvision::common::sdk::ConfigurationProperty
//------------------------------------------------------------------------------
class EnumConfigurationProperty : public virtual ConfigurationProperty
{
protected:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::EnumConfigurationProperty";

    //========================================
    //! \brief Provides common logger interface.
    //! \returns Smart pointer to microvision::common::logging::Logger.
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr getLogger()
    {
        static auto logger{microvision::common::logging::LogManager::getInstance().createLogger(m_loggerId)};
        return logger;
    }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    EnumConfigurationProperty() : ConfigurationProperty() {}

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    EnumConfigurationProperty(const EnumConfigurationProperty& other) : ConfigurationProperty(other) {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~EnumConfigurationProperty() = default;

public:
    //========================================
    //! \brief Get enum value type.
    //! \return Enum value type info.
    //----------------------------------------
    virtual const std::type_info& getEnumType() const = 0;

    //========================================
    //! \brief Get underlying value type.
    //! \return Underlying value type info.
    //----------------------------------------
    virtual const std::type_info& getUnderlyingType() const = 0;

    //========================================
    //! \brief Get all possible enum names.
    //! \return All possible enum names.
    //----------------------------------------
    virtual const std::vector<std::string>& getPossibleNames() const = 0;

public: // getter
    //========================================
    //! \brief Get name of the enum value.
    //! \returns Name of the enum value.
    //----------------------------------------
    virtual Optional<std::string> getNameByValue() const = 0;

    //========================================
    //! \brief Get name of the default value.
    //! \returns Name of the default value.
    //----------------------------------------
    virtual std::string getNameByDefaultValue() const = 0;

    //========================================
    //! \brief Get Name of (default) value.
    //! \returns Name of (Default) value.
    //----------------------------------------
    virtual std::string getNameByValueOrDefault() const = 0;

public: // setter
    //========================================
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    virtual void setValueByName(const std::string& value) = 0;

    //========================================
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    virtual void setValueByName(std::string&& value) = 0;

    //========================================
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    virtual void setValueByName(const Optional<std::string>& value) = 0;

    //========================================
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    virtual void setValueByName(Optional<std::string>&& value) = 0;

}; // class EnumConfigurationProperty

//==============================================================================
//! \brief Abstract base implementation of enum configurable property.
//! \tparam ValueType  Enum value type.
//! \extends microvision::common::sdk::EnumConfigurationProperty
//------------------------------------------------------------------------------
template<typename ValueType, typename = EnableIfEnum<ValueType>>
class EnumConfigurationPropertyOf : public virtual EnumConfigurationProperty
{
public:
    //========================================
    //! \brief Enum underlying type.
    //----------------------------------------
    using UnderlyingType = typename std::underlying_type<ValueType>::type;

    //========================================
    //! \brief Map of all value name enum pairs.
    //----------------------------------------
    using ValueNameMap = std::map<ValueType, std::string>;

    //========================================
    //! \brief Map of all name value enum pairs.
    //----------------------------------------
    using NameValueMap = std::map<std::string, ValueType>;

    //========================================
    //! \brief Initializer list of value name enum pairs.
    //----------------------------------------
    using ValueNameInitList = std::initializer_list<typename ValueNameMap::value_type>;

protected:
    //========================================
    //! \brief Empty constructor to fix virtual inheritance tree instantiation.
    //! \notes Should never used in runtime, is just to fix compile checks.
    //----------------------------------------
    EnumConfigurationPropertyOf()
      : ConfigurationProperty(),
        EnumConfigurationProperty(),
        m_value{},
        m_defaultValue{},
        m_valueNameMap{},
        m_nameValueMap{},
        m_names{}
    {}

public:
    //========================================
    //! \brief Construct configurable property with default value and enum name value map.
    //! \param[in] defaultValue     Default value.
    //! \param[in] valueNames       Enum name value map.
    //----------------------------------------
    EnumConfigurationPropertyOf(const ValueType defaultValue, ValueNameInitList valueNames)
      : ConfigurationProperty(),
        EnumConfigurationProperty(),
        m_value{},
        m_defaultValue{defaultValue},
        m_valueNameMap{valueNames},
        m_nameValueMap{},
        m_names{}
    {
        for (const auto& valueNameEntry : this->m_valueNameMap)
        {
            this->m_nameValueMap.insert(std::make_pair(valueNameEntry.second, valueNameEntry.first));
            this->m_names.push_back(valueNameEntry.second);
        }
    }

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    EnumConfigurationPropertyOf(const EnumConfigurationPropertyOf& other)
      : ConfigurationProperty(other),
        EnumConfigurationProperty(other),
        m_value{other.m_value.getValue()},
        m_defaultValue{other.m_defaultValue},
        m_valueNameMap{other.m_valueNameMap},
        m_nameValueMap{other.m_nameValueMap},
        m_names{other.m_names}
    {}

    //========================================
    //! \brief Move constructor (deleted).
    //----------------------------------------
    EnumConfigurationPropertyOf(EnumConfigurationPropertyOf&&) noexcept = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~EnumConfigurationPropertyOf() = default;

public: // implements
    //========================================
    //! \brief Get enum value type.
    //! \return Enum value type info.
    //----------------------------------------
    const std::type_info& getEnumType() const override { return typeid(ValueType); }

    //========================================
    //! \brief Get underlying value type.
    //! \return Underlying value type info.
    //----------------------------------------
    const std::type_info& getUnderlyingType() const override { return typeid(UnderlyingType); }

    //========================================
    //! \brief Get all possible enum names.
    //! \return All possible enum names.
    //----------------------------------------
    const std::vector<std::string>& getPossibleNames() const override { return this->m_names; }

public: // implements ConfigurationPropertyOf<std::string>
    //========================================
    //! \brief Copy configuration property value from another configuration property.
    //! \param[in] other  Other configuration property to copy value from.
    //! \return Either \c true if value could been copied, otherwise \c false if not.
    //----------------------------------------
    bool copyValueFrom(const ConfigurationProperty& other) override
    {
        const auto property = dynamic_cast<const EnumConfigurationPropertyOf<ValueType>*>(&other);

        if (property != nullptr)
        {
            this->setValueByName(property->getNameByValue());
            return true;
        }
        return false;
    }

    //========================================
    //! \brief Get name of value.
    //! \returns Name of value.
    //----------------------------------------
    Optional<std::string> getNameByValue() const override
    {
        auto valueAccess = this->m_value.get();

        if (*valueAccess)
        {
            const auto it = this->m_valueNameMap.find(**valueAccess);

            if (it != this->m_valueNameMap.end())
            {
                return Optional<std::string>{it->second};
            }
            else
            {
                LOGWARNING(getLogger(),
                           "Enum value " << toHex(static_cast<UnderlyingType>(**valueAccess))
                                         << " is not mapped for configuration property " << this->getId());
            }
        }

        return Optional<std::string>{nullopt};
    }

    //========================================
    //! \brief Get name of default value.
    //! \returns Name of Default value.
    //----------------------------------------
    std::string getNameByDefaultValue() const override
    {
        auto it = this->m_valueNameMap.find(this->m_defaultValue);

        if (it != this->m_valueNameMap.end())
        {
            return it->second;
        }
        else
        {
            LOGWARNING(getLogger(),
                       "Enum default value " << toHex(static_cast<UnderlyingType>(this->m_defaultValue))
                                             << " is not mapped for configuration property " << this->getId());
        }

        return std::string{};
    }

    //========================================
    //! \brief Get name of (default) value.
    //! \returns Name of (default) value.
    //----------------------------------------
    std::string getNameByValueOrDefault() const override
    {
        auto value = this->getNameByValue();

        if (value)
        {
            return *value;
        }

        return this->getNameByDefaultValue();
    }

    //========================================
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    void setValueByName(const std::string& value) override { this->setValueByName(Optional<std::string>{value}); }

    //========================================
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    void setValueByName(std::string&& value) override { this->setValueByName(Optional<std::string>{std::move(value)}); }

    //========================================
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    void setValueByName(const Optional<std::string>& value) override
    {
        Any oldValue{};
        {
            auto valueAccess = m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                const auto it = this->m_valueNameMap.find(**valueAccess);

                if (it != this->m_valueNameMap.end())
                {
                    oldValue = it->second;
                }
                else
                {
                    LOGWARNING(getLogger(),
                               "Enum value " << toHex(static_cast<UnderlyingType>(**valueAccess))
                                             << " is not mapped for configuration property " << this->getId());
                }
            }

            if (value)
            {
                const auto it = this->m_nameValueMap.find(*value);

                if (it != this->m_nameValueMap.end())
                {
                    valueAccess = Optional<ValueType>{it->second};
                }
                else
                {
                    LOGWARNING(getLogger(),
                               "Enum value " << *value << " is not mapped for configuration property "
                                             << this->getId());
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
    //! \brief Set value by name.
    //! \param[in] value  New name of value.
    //----------------------------------------
    void setValueByName(Optional<std::string>&& value) override
    {
        Any oldValue{};
        {
            auto valueAccess = m_value.get();

            if (*valueAccess && this->hasOnSetValueEventListener())
            {
                const auto it = this->m_valueNameMap.find(**valueAccess);

                if (it != this->m_valueNameMap.end())
                {
                    oldValue = it->second;
                }
                else
                {
                    LOGWARNING(getLogger(),
                               "Enum value " << toHex(static_cast<UnderlyingType>(**valueAccess))
                                             << " is not mapped for configuration property " << this->getId());
                }
            }

            if (value)
            {
                const auto it = this->m_nameValueMap.find(*value);

                if (it != this->m_nameValueMap.end())
                {
                    valueAccess = Optional<ValueType>{it->second};
                }
                else
                {
                    LOGWARNING(getLogger(),
                               "Enum value " << *value << " is not mapped for configuration property "
                                             << this->getId());
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

public:
    //========================================
    //! \brief Get map of all values and names.
    //! \return Value name map.
    //----------------------------------------
    const ValueNameMap& getValueNameMap() const { return this->m_valueNameMap; }

    //========================================
    //! \brief Get map of all names and values.
    //! \return Name value map.
    //----------------------------------------
    const NameValueMap& getNameValueMap() const { return this->m_nameValueMap; }

protected:
    //========================================
    //! \brief Value of property.
    //----------------------------------------
    ThreadSafe<Optional<ValueType>> m_value;

    //========================================
    //! \brief Default value of property.
    //----------------------------------------
    ValueType m_defaultValue;

    //========================================
    //! \brief Map of all values and names.
    //----------------------------------------
    ValueNameMap m_valueNameMap;

    //========================================
    //! \brief Map of all names and values.
    //----------------------------------------
    NameValueMap m_nameValueMap;

    //========================================
    //! \brief List of all names.
    //----------------------------------------
    std::vector<std::string> m_names;

}; // class EnumConfigurationPropertyOf

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
