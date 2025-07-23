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

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/Utils.hpp>

#include <microvision/common/logging/logging.hpp>

#include <type_traits>
#include <stdexcept>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Exception thrown if cast of any reference failed.
//------------------------------------------------------------------------------
class BadAnyCast final : public std::bad_cast
{
public:
    //========================================
    //! \brief Constructs exception and message with source and target type arguments.
    //! \param[in] fromType  Source type of any value.
    //! \param[in] toType    Target type of cast.
    //----------------------------------------
    BadAnyCast(const std::type_info& fromType, const std::type_info& toType)
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
    ~BadAnyCast() override = default;

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
};

//==============================================================================
//! \brief The class any describes a type-safe container for single values of any type.
//!
//! \note We do not use the boost::any because of an issue with the assignment operator
//!       of value which not exclude boost::any in boost version <= 1.65.1.
//------------------------------------------------------------------------------
class Any final
{
private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::any";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr logger()
    {
        static microvision::common::logging::LoggerSPtr log{
            microvision::common::logging::LogManager::getInstance().createLogger(Any::m_loggerId)};
        return log;
    }

private:
    //========================================
    //! \brief Abstract value container base for generic storage.
    //----------------------------------------
    class AnyValue
    {
    public:
        //========================================
        //! \brief Use the default destructor.
        //----------------------------------------
        virtual ~AnyValue() = default;

    public:
        //========================================
        //! \brief Get type info of value type.
        //! \returns Value type info.
        //----------------------------------------
        virtual const std::type_info& type() const = 0;

        //========================================
        //! \brief Get copy of value container.
        //! \returns Copy of this instance.
        //----------------------------------------
        virtual std::unique_ptr<AnyValue> copy() const = 0;
    }; // AnyValue

    //========================================
    //! \brief Implementation of generic value container to store any value.
    //----------------------------------------
    template<typename ValueType>
    class AnyValueImpl final : public AnyValue
    {
    public:
        //========================================
        //! \brief Empty construction.
        //----------------------------------------
        AnyValueImpl() : value{} {}

        //========================================
        //! \brief Move construction.
        //! \param[in] val  Value to move.
        //----------------------------------------
        AnyValueImpl(ValueType&& val) noexcept : value{std::move(val)} {}

        //========================================
        //! \brief Copy construction.
        //! \param[in] val  Value to copy.
        //----------------------------------------
        AnyValueImpl(const ValueType& val) : value{val} {}

        //========================================
        //! \brief Use the default destructor.
        //----------------------------------------
        ~AnyValueImpl() override = default;

    public:
        //========================================
        //! \brief Get type info of template argument ValueType.
        //! \returns Value type info.
        //----------------------------------------
        const std::type_info& type() const override { return typeid(ValueType); }

        //========================================
        //! \brief Get copy of value container implementation.
        //! \returns Copy of this instance.
        //----------------------------------------
        std::unique_ptr<AnyValue> copy() const override
        {
            return std::make_unique<AnyValueImpl<ValueType>>(this->value);
        }

    public:
        //========================================
        //! \brief Value store.
        //----------------------------------------
        ValueType value;
    }; // AnyValueImpl

public:
    //========================================
    //! \brief Conditional check that the value type is not any.
    //!
    //! This is needed to prevent that any contains any.
    //----------------------------------------
    template<typename ValueType>
    using EnableIfNotany = std::enable_if<!std::is_same<Any, std::remove_cvref<ValueType>>::value, int>;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    constexpr Any() noexcept : m_anyValue{} {}

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another any from which will takeover the value.
    //----------------------------------------
    Any(Any&& other) noexcept : m_anyValue{std::move(other.m_anyValue)} {}

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another any from which will copied the value.
    //----------------------------------------
    Any(const Any& other) : m_anyValue{}
    {
        if (other.has_value())
        {
            this->m_anyValue = other.m_anyValue->copy();
        }
    }

    //========================================
    //! \brief Implicit constructor for any value, except any itself.
    //! \param[in] value  any value from which will takeover the value.
    //----------------------------------------
    template<typename ValueType, typename Any::EnableIfNotany<ValueType>::type = 0>
    explicit Any(ValueType&& value) noexcept
      : m_anyValue{std::make_unique<AnyValueImpl<std::remove_cvref<ValueType>>>(std::move(value))}
    {}

    //========================================
    //! \brief Implicit constructor for any value, except any itself.
    //! \param[in] value  any value from which will copied the value.
    //----------------------------------------
    template<typename ValueType, typename Any::EnableIfNotany<ValueType>::type = 0>
    explicit Any(const ValueType& value)
      : m_anyValue{std::make_unique<AnyValueImpl<std::remove_cvref<ValueType>>>(value)}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Any() = default;

public:
    //========================================
    //! \brief Move assigment operator.
    //! \param[in] other  Another any from which will takeover the value.
    //! \returns Reference of this.
    //----------------------------------------
    Any& operator=(Any&& other) noexcept
    {
        this->m_anyValue = std::move(other.m_anyValue);
        return *this;
    }

    //========================================
    //! \brief Copy assigment operator.
    //! \param[in] other  Another any from which will copied the value.
    //! \returns Reference of this.
    //----------------------------------------
    Any& operator=(const Any& other)
    {
        if (other.has_value())
        {
            this->m_anyValue = other.m_anyValue->copy();
        }
        else
        {
            this->reset();
        }
        return *this;
    }

    //========================================
    //! \brief Implicit move assignment operator for any value, except any itself.
    //! \param[in] value  any value from which will takeover the value.
    //! \returns Reference of this.
    //----------------------------------------
    template<typename ValueType, typename Any::EnableIfNotany<ValueType>::type = 0>
    Any& operator=(ValueType&& value) noexcept
    {
        this->m_anyValue.reset(new AnyValueImpl<std::remove_cvref<ValueType>>{std::move(value)});
        return *this;
    }

    //========================================
    //! \brief Implicit copy assignment operator for any value, except any itself.
    //! \param[in] value  any value from which will copied the value.
    //! \returns Reference of this.
    //----------------------------------------
    template<typename ValueType, typename Any::EnableIfNotany<ValueType>::type = 0>
    Any& operator=(const ValueType& value)
    {
        this->m_anyValue.reset(new AnyValueImpl<std::remove_cvref<ValueType>>{value});
        return *this;
    }

public:
    //========================================
    //! \brief Checks if value is set.
    //! \returns Either \c true if value is set or otherwise \c false.
    //----------------------------------------
    bool has_value() const noexcept { return this->m_anyValue != nullptr; }

    //========================================
    //! \brief Get the type info of the containing value.
    //! \note To compare the type info use '...type() == typeid(%VALUE_TYPE%)...'.
    //! \returns Either if value is set the type info of value, otherwise the type info of void.
    //----------------------------------------
    const std::type_info& type() const noexcept
    {
        if (this->has_value())
        {
            return this->m_anyValue->type();
        }
        else
        {
            return typeid(void);
        }
    }

    //========================================
    //! \brief Exchange the values of this and the \a other any container.
    //! \param[in/out] other  Another any container to swap the values.
    //! \returns Reference of this.
    //----------------------------------------
    Any& swap(Any& other) noexcept
    {
        this->m_anyValue.swap(other.m_anyValue);
        return *this;
    }

    //========================================
    //! \brief Removed the value.
    //----------------------------------------
    void reset() noexcept { this->m_anyValue.reset(); }

public:
    //========================================
    //! \brief Get private access for value cast.
    //----------------------------------------
    template<typename T>
    friend T& anyCast(Any& anyValue);

    //========================================
    //! \brief Get private access for value cast.
    //----------------------------------------
    template<typename T>
    friend const T& anyCast(const Any& anyValue);

    //========================================
    //! \brief Get private access for value cast.
    //----------------------------------------
    template<typename T>
    friend T* anyCast(Any* anyValue) noexcept;

    //========================================
    //! \brief Get private access for value cast.
    //----------------------------------------
    template<typename T>
    friend const T* anyCast(const Any* anyValue) noexcept;

private:
    //========================================
    //! \brief Pointer to any value store.
    //----------------------------------------
    std::unique_ptr<AnyValue> m_anyValue;
};

//==============================================================================

//========================================
//! \brief Cast any reference to value of type T.
//! \tparam T  Type of any value.
//! \param[in] AnyValue  any value container.
//! \throws std::invalid_argument  Thrown if parameter AnyValue is empty.
//! \throws BadAnyCast             Thrown if cast of value to type T failed.
//! \returns Reference to value of type T which is contained by any.
//----------------------------------------
template<typename T>
T& anyCast(Any& anyValue)
{
    if (!anyValue.has_value())
    {
        throw std::invalid_argument{"any value is empty."};
    }

    T* val = anyCast<T>(&anyValue);

    if (val == nullptr)
    {
        throw BadAnyCast(anyValue.type(), typeid(T));
    }

    return *val;
}

//========================================
//! \brief Cast any reference to value of type T.
//! \tparam T  Type of any value.
//! \param[in] anyValue  any value container.
//! \throws std::invalid_argument  Thrown if parameter anyValue is empty.
//! \throws BadAnyCast             Thrown if cast of value to type T failed.
//! \returns Reference to value of type T which is contained by any.
//----------------------------------------
template<typename T>
const T& anyCast(const Any& anyValue)
{
    if (!anyValue.has_value())
    {
        throw std::invalid_argument{"any value is empty."};
    }

    const T* val = anyCast<T>(&anyValue);

    if (val == nullptr)
    {
        throw BadAnyCast(anyValue.type(), typeid(T));
    }

    return *val;
}

//========================================
//! \brief Cast any pointer to value of type T.
//! \tparam T  Type of any value.
//! \param[in] anyValue  any value container.
//! \returns Either pointer of value type T if cast is successful or otherwise nullptr.
//----------------------------------------
template<typename T>
T* anyCast(Any* anyValue) noexcept
{
    if ((anyValue == nullptr) || !anyValue->has_value())
    {
        LOGWARNING(Any::logger(), "any value is empty.");
        return nullptr;
    }

    Any::AnyValueImpl<T>* val = dynamic_cast<Any::AnyValueImpl<T>*>(anyValue->m_anyValue.get());

    if (val == nullptr)
    {
        return nullptr;
    }

    return &val->value;
}

//========================================
//! \brief Cast any pointer to value of type T.
//! \tparam T  Type of any value.
//! \param[in] anyValue  any value container.
//! \returns Either pointer of value type T if cast is successful or otherwise nullptr.
//----------------------------------------
template<typename T>
const T* anyCast(const Any* anyValue) noexcept
{
    if ((anyValue == nullptr) || !anyValue->has_value())
    {
        LOGWARNING(Any::logger(), "any value is empty.");
        return nullptr;
    }

    Any::AnyValueImpl<T>* val = dynamic_cast<Any::AnyValueImpl<T>*>(anyValue->m_anyValue.get());

    if (val == nullptr)
    {
        return nullptr;
    }

    return &val->value;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
