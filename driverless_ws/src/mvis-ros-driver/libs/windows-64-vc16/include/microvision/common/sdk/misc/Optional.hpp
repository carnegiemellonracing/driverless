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
//! \date Jan 21, 2020
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <limits>

#include <microvision/common/sdk/misc/Utils.hpp>

#include <type_traits>
#include <stdexcept>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Exception that is thrown when accessing an optional object that does not contain a value.
//------------------------------------------------------------------------------
class BadOptionalAccess final : public std::logic_error
{
public:
    //========================================
    //! \brief Constructs exception and message.
    //----------------------------------------
    BadOptionalAccess() : std::logic_error{"Optional value is not set."} {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BadOptionalAccess() override = default;

}; // class BadOptionalAccess

//==============================================================================
//! \brief Indicator of optional type with uninitialized state.
//------------------------------------------------------------------------------
using nullopt_t = std::nullptr_t;

//==============================================================================
//! \brief This is a constant of type nullopt_t that is used to indicate optional type with uninitialized state.
//------------------------------------------------------------------------------
constexpr nullopt_t nullopt = nullptr;

//==============================================================================
//! \brief The class template Optional manages an optional contained value,
//!        i.e. a value that may or may not be present.
//!
//! \tparam ValueType  Any kind of value type.
//------------------------------------------------------------------------------
template<typename ValueType>
class Optional final
{
public:
    //========================================
    //! \brief The type of the value to manage initialization state for.
    //----------------------------------------
    using value_type = std::remove_cvref<ValueType>;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    constexpr Optional() noexcept {}

    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    constexpr Optional(const nullopt_t) noexcept {}

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another Optional from which will takeover the value.
    //----------------------------------------
    Optional(Optional&& other) noexcept
    {
        if (other.has_value())
        {
            this->constructValue(std::move(other.value()));
        }
        other.destructValue();
    }

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another Optional from which will copied the value.
    //----------------------------------------
    Optional(const Optional& other)
    {
        if (other.has_value())
        {
            this->constructValue(other.value());
        }
    }

    //========================================
    //! \brief Implicit move constructor for value.
    //! \param[in] value  Value from which will takeover the value.
    //----------------------------------------
    explicit Optional(value_type&& value) { this->constructValue(std::move(value)); }

    //========================================
    //! \brief Implicit copy constructor for value.
    //! \param[in] value  Value from which the value will copied.
    //----------------------------------------
    explicit Optional(const value_type& value) { this->constructValue(value); }

    //========================================
    //! \brief Default destructor.
    //!
    //! Contained value will as well destructed.
    //----------------------------------------
    ~Optional() { this->destructValue(); };

public:
    //========================================
    //! \brief Empty assigment operator.
    //! \return Reference of this.
    //----------------------------------------
    Optional& operator=(nullopt_t) noexcept
    {
        this->destructValue();
        return *this;
    }

    //========================================
    //! \brief Implicit copy assignment operator for value.
    //! \param[in] value  Value from which the value will be taken.
    //! \return Reference of this.
    //----------------------------------------
    Optional& operator=(const value_type& value)
    {
        this->constructValue(value);
        return *this;
    }

    //========================================
    //! \brief Implicit move assignment operator for value.
    //! \param[in] value  Value from which the value will be taken.
    //! \returns Reference of this.
    //----------------------------------------
    Optional& operator=(value_type&& value)
    {
        this->constructValue(std::move(value));
        return *this;
    }

    //========================================
    //! \brief Move assigment operator.
    //! \param[in] other  Another Optional value from which will takeover the value.
    //! \returns Reference of this.
    //----------------------------------------
    Optional& operator=(Optional&& other) noexcept
    {
        if (other.has_value())
        {
            this->constructValue(std::move(other.value()));
        }
        else
        {
            this->destructValue();
        }
        other.destructValue();
        return *this;
    }

    //========================================
    //! \brief Copy assigment operator.
    //! \param[in] other  Another optional value from which will copied the value.
    //! \returns Reference of this.
    //----------------------------------------
    Optional& operator=(const Optional& other)
    {
        if (other.has_value())
        {
            this->constructValue(other.value());
        }
        else
        {
            this->destructValue();
        }
        return *this;
    }

public:
    //========================================
    //! \brief Get pointer to contained value.
    //! \returns Either \c value pointer if set or otherwise \c nullptr.
    //----------------------------------------
    value_type* operator->() { return this->m_value; }

    //========================================
    //! \brief Get pointer to contained value.
    //! \returns Either \c value pointer if set or otherwise \c nullptr.
    //----------------------------------------
    const value_type* operator->() const { return this->m_value; }

    //========================================
    //! \brief Get reference to contained value.
    //! \exception BadOptionalAccess  Will be thrown if value is not set.
    //! \returns Either \c value reference if set or otherwise will an \c exception thrown.
    //----------------------------------------
    value_type& operator*() { return this->value(); }

    //========================================
    //! \brief Get reference to contained value.
    //! \exception BadOptionalAccess  Will be thrown if value is not set.
    //! \returns Either \c value reference if set or otherwise will an \c exception thrown.
    //----------------------------------------
    const value_type& operator*() const { return this->value(); }

public:
    explicit operator bool() const noexcept { return this->has_value(); }

    //========================================
    //! \brief Checks if value is set.
    //! \returns Either \c true if value is set or otherwise \c false.
    //----------------------------------------
    bool has_value() const noexcept { return this->m_value != nullptr; }

public:
    //========================================
    //! \brief Get reference to contained value.
    //! \exception BadOptionalAccess  Will be thrown if value is not set.
    //! \returns Either \c value reference if set or otherwise will an \c exception thrown.
    //----------------------------------------
    value_type& value()
    {
        value_type* val = this->m_value;

        if (val == nullptr)
        {
            throw BadOptionalAccess();
        }
        return *val;
    }

    //========================================
    //! \brief Get reference to contained value.
    //! \exception BadOptionalAccess  Will be thrown if value is not set.
    //! \returns Either \c value reference if set or otherwise will an \c exception thrown.
    //----------------------------------------
    const value_type& value() const
    {
        const value_type* val = this->m_value;

        if (val == nullptr)
        {
            throw BadOptionalAccess();
        }
        return *val;
    }

    //========================================
    //! \brief Get copy of contained value or default value.
    //! \param[in] defaultValue  The default value which will returned if no value is set.
    //! \returns Either value copy if set or otherwise the default value.
    //----------------------------------------
    value_type value_or(value_type&& defaultValue) const
    {
        const value_type* val = this->m_value;

        if (val == nullptr)
        {
            return std::move(defaultValue);
        }
        return *val;
    }

    //========================================
    //! \brief Exchange the values of this and the \a other Optional container.
    //! \param[in/out] other  Another Optional container to swap the values.
    //! \returns Reference of this.
    //----------------------------------------
    Optional& swap(Optional& other) noexcept
    {
        if (this->has_value() && other.has_value())
        {
            value_type tmpValue = std::move(other.value());
            other.constructValue(std::move(this->value()));
            this->constructValue(std::move(tmpValue));
        }
        else if (this->has_value())
        {
            this->destructValue();
        }
        else if (other.has_value())
        {
            this->constructValue(std::move(other.value()));
            other.destructValue();
        }
        return *this;
    }

    //========================================
    //! \brief Removed the value.
    //----------------------------------------
    void reset() noexcept { this->destructValue(); }

    //==============================================================================
    //!\brief Compares two floats \a a and \a b. NaN is equal NaN here.
    //!\param[in] a  First float to be compared.
    //!\param[in] b  Second float to be compared.
    //!\return \c true if \a a == \a b or if both are NaN.
    //!        \c false otherwise.
    //------------------------------------------------------------------------------
    static bool isEqual(const float& value1, const float& value2)
    {
        return ((std::isnan(value1) && std::isnan(value2)) || std::fabs(value1 - value2) < 1E-12F);
    }

    //==============================================================================
    //!\brief Compares two double \a a and \a b. NaN is equal NaN here.
    //!\param[in] a  First double to be compared.
    //!\param[in] b  Second double to be compared.
    //!\return \c true if \a a == \a b or if both are NaN.
    //!        \c false otherwise.
    //------------------------------------------------------------------------------
    static bool isEqual(const double& value1, const double& value2)
    {
        return ((std::isnan(value1) && std::isnan(value2)) || static_cast<double>(std::fabs(value1 - value2)) < 1E-17);
    }

    //==============================================================================
    //!\brief Compares two values \a a and \a b.
    //!\param[in] a  First values to be compared.
    //!\param[in] b  Second values to be compared.
    //!\return \c true if \a a == \a b.
    //!        \c false otherwise.
    //------------------------------------------------------------------------------
    template<typename T_ = ValueType, typename std::enable_if<!std::is_floating_point<T_>::value, int>::type = 0>
    static bool isEqual(const T_ a, const T_ b)
    {
        return a == b;
    }

public:
    //========================================
    //! \brief Compares the Optional values on equality.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const Optional& lhs, const Optional& rhs)
    {
        return (lhs.has_value() && rhs.has_value() && isEqual(lhs.value(), rhs.value()))
               || (!lhs.has_value() && !rhs.has_value());
    }

    //========================================
    //! \brief Compares the Optional values on inequality.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const Optional& lhs, const Optional& rhs)
    {
        return (lhs.has_value() && rhs.has_value() && (!isEqual(lhs.value(), rhs.value())))
               || ((!lhs.has_value() && rhs.has_value()) || (lhs.has_value() && !rhs.has_value()));
    }

    //========================================
    //! \brief Checks if left Optional value is smaller than right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if left value is smaller than right value or otherwise \c false.
    //----------------------------------------
    friend bool operator<(const Optional& lhs, const Optional& rhs)
    {
        return (lhs.has_value() && rhs.has_value() && lhs.value() < rhs.value())
               || ((!lhs.has_value() && rhs.has_value()));
    }

    //========================================
    //! \brief Checks if left Optional value is smaller or equals to right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if left value is smaller or equals to right value or otherwise \c false.
    //----------------------------------------
    friend bool operator<=(const Optional& lhs, const Optional& rhs)
    {
        return (lhs.has_value() && rhs.has_value() && lhs.value() <= rhs.value())
               || ((!lhs.has_value() && rhs.has_value()) || (!lhs.has_value() && !rhs.has_value()));
    }

    //========================================
    //! \brief Checks if left Optional value is greater than right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if left value is greater than right value or otherwise \c false.
    //----------------------------------------
    friend bool operator>(const Optional& lhs, const Optional& rhs)
    {
        return (lhs.has_value() && rhs.has_value() && lhs.value() > rhs.value())
               || (lhs.has_value() && !rhs.has_value());
    }

    //========================================
    //! \brief Checks if left Optional value is greater or equals to right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if left value is greater or equals to right value or otherwise \c false.
    //----------------------------------------
    friend bool operator>=(const Optional& lhs, const Optional& rhs)
    {
        return (lhs.has_value() && rhs.has_value() && lhs.value() >= rhs.value())
               || ((lhs.has_value() && !rhs.has_value()) || (!lhs.has_value() && !rhs.has_value()));
    }

public:
    //========================================
    //! \brief Checks if the left Optional value is not set.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as nullopt_t constant nullopt.
    //! \returns Either \c true if left value is not set or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const Optional& lhs, nullopt_t) noexcept { return !lhs.has_value(); }

    //========================================
    //! \brief Checks if the left Optional value is set.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as nullopt_t constant nullopt.
    //! \returns Either \c true if left value is set or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const Optional& lhs, nullopt_t) noexcept { return lhs.has_value(); }

    //========================================
    //! \brief Checks if the left Optional value is not set.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as nullopt_t constant nullopt.
    //! \returns Either \c true if left value is not set or otherwise \c false.
    //----------------------------------------
    friend bool operator<(const Optional& lhs, nullopt_t) noexcept { return !lhs.has_value(); }

    //========================================
    //! \brief Checks if the left Optional value is not set.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as nullopt_t constant nullopt.
    //! \returns Either \c true if left value is not set or otherwise \c false.
    //----------------------------------------
    friend bool operator<=(const Optional& lhs, nullopt_t) noexcept { return !lhs.has_value(); }

    //========================================
    //! \brief Checks if the left Optional value is set.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as nullopt_t constant nullopt.
    //! \returns Either \c true if left value is set or otherwise \c false.
    //----------------------------------------
    friend bool operator>(const Optional& lhs, nullopt_t) noexcept { return lhs.has_value(); }

    //========================================
    //! \brief Checks nothing.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as nullopt_t constant nullopt.
    //! \returns \c true
    //----------------------------------------
    friend bool operator>=(const Optional&, nullopt_t) noexcept { return true; }

public:
    //========================================
    //! \brief Checks if the right Optional value is not set.
    //! \param[in] lhs  Left operator value, as nullopt_t constant nullopt.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right value is not set or otherwise \c false.
    //----------------------------------------
    friend bool operator==(nullopt_t, const Optional& rhs) noexcept { return !rhs.has_value(); }

    //========================================
    //! \brief Checks if the right Optional value is set.
    //! \param[in] lhs  Left operator value, as nullopt_t constant nullopt.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right value is set or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(nullopt_t, const Optional& rhs) noexcept { return rhs.has_value(); }

    //========================================
    //! \brief Checks if the right Optional value is not set.
    //! \param[in] lhs  Left operator value, as nullopt_t constant nullopt.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right value is not set or otherwise \c false.
    //----------------------------------------
    friend bool operator<(nullopt_t, const Optional& rhs) noexcept { return rhs.has_value(); }

    //========================================
    //! \brief Checks if the right Optional value is not set.
    //! \param[in] lhs  Left operator value, as nullopt_t constant nullopt.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right value is not set or otherwise \c false.
    //----------------------------------------
    friend bool operator<=(nullopt_t, const Optional&) noexcept { return true; }

    //========================================
    //! \brief Checks if the right Optional value is set.
    //! \param[in] lhs  Left operator value, as nullopt_t constant nullopt.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right value is set or otherwise \c false.
    //----------------------------------------
    friend bool operator>(nullopt_t, const Optional& rhs) noexcept { return !rhs.has_value(); }

    //========================================
    //! \brief Checks nothing.
    //! \param[in] lhs  Left operator value, as nullopt_t constant nullopt.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns \c true
    //----------------------------------------
    friend bool operator>=(nullopt_t, const Optional& rhs) noexcept { return !rhs.has_value(); }

public:
    //========================================
    //! \brief Checks whether the left Optional value is set
    //!        and the left and right value is equals.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as value_type instance.
    //! \returns Either \c true if left Optional value is set
    //!          and the left and right value is equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const Optional& lhs, const value_type& rhs)
    {
        return lhs.has_value() && isEqual(lhs.value(), rhs);
    }

    //========================================
    //! \brief Checks whether the left Optional value is not set
    //!        or the left and right value is inequals.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as value_type instance.
    //! \returns Either \c true if left Optional value is not set
    //!          or the left and right value is inequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const Optional& lhs, const value_type& rhs)
    {
        return !lhs.has_value() || (!isEqual(lhs.value(), rhs));
    }

    //========================================
    //! \brief Checks whether the left Optional value is not set
    //!        or the left value is smaller than the right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as value_type instance.
    //! \returns Either \c true if left Optional value is not set
    //!          or the left value is smaller than the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator<(const Optional& lhs, const value_type& rhs) { return !lhs.has_value() || lhs.value() < rhs; }

    //========================================
    //! \brief Checks whether the left Optional value is not set
    //!        or the left value is smaller or equals to the right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as value_type instance.
    //! \returns Either \c true if left Optional value is not set
    //!          or the left value is smaller or equals to the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator<=(const Optional& lhs, const value_type& rhs)
    {
        return !lhs.has_value() || lhs.value() <= rhs;
    }

    //========================================
    //! \brief Checks whether the left Optional value is set
    //!        and the left value is greater than the right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as value_type instance.
    //! \returns Either \c true if left Optional value is set
    //!          and the left value is greater than the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator>(const Optional& lhs, const value_type& rhs) { return lhs.has_value() && lhs.value() > rhs; }

    //========================================
    //! \brief Checks whether the left Optional value is set
    //!        and the left value is greater or equals to the right value.
    //! \param[in] lhs  Left operator value, as Optional instance.
    //! \param[in] rhs  Right operator value, as value_type instance.
    //! \returns Either \c true if left Optional value is set
    //!          and the left value is greater or equals to the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator>=(const Optional& lhs, const value_type& rhs) { return lhs.has_value() && lhs.value() >= rhs; }

public:
    //========================================
    //! \brief Checks whether the right Optional value is set
    //!        and the left is equals to the right value.
    //! \param[in] lhs  Left operator value, as value_type instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right Optional value is set
    //!          and the left value is equals to the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const value_type& lhs, const Optional& rhs)
    {
        return rhs.has_value() && isEqual(lhs, rhs.value());
    }

    //========================================
    //! \brief Checks whether the right Optional value is not set
    //!        or the left is inequals to the right value.
    //! \param[in] lhs  Left operator value, as value_type instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right Optional value is not set
    //!          or the left value is inequals to the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const value_type& lhs, const Optional& rhs)
    {
        return !rhs.has_value() || (!isEqual(lhs, rhs.value()));
    }

    //========================================
    //! \brief Checks whether the right Optional value is set
    //!        and the left is smaller than the right value.
    //! \param[in] lhs  Left operator value, as value_type instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right Optional value is set
    //!          and the left value is smaller than the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator<(const value_type& lhs, const Optional& rhs) { return rhs.has_value() && lhs < rhs.value(); }

    //========================================
    //! \brief Checks whether the right Optional value is set
    //!        and the left is smaller or equals to the right value.
    //! \param[in] lhs  Left operator value, as value_type instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right Optional value is set
    //!          and the left value is smaller or equals the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator<=(const value_type& lhs, const Optional& rhs) { return rhs.has_value() && lhs <= rhs.value(); }

    //========================================
    //! \brief Checks whether the right Optional value is not set
    //!        or the left is greater than the right value.
    //! \param[in] lhs  Left operator value, as value_type instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right Optional value is not set
    //!          and the left value is greater than the right value or otherwise \c false.
    //----------------------------------------
    friend bool operator>(const value_type& lhs, const Optional& rhs) { return !rhs.has_value() || lhs > rhs.value(); }

    //========================================
    //! \brief Checks whether the right Optional value is not set
    //!        or the left is greater or equals to right value.
    //! \param[in] lhs  Left operator value, as value_type instance.
    //! \param[in] rhs  Right operator value, as Optional instance.
    //! \returns Either \c true if right Optional value is not set
    //!          and the left value is greater or equals to right value or otherwise \c false.
    //----------------------------------------
    friend bool operator>=(const value_type& lhs, const Optional& rhs)
    {
        return !rhs.has_value() || lhs >= rhs.value();
    }

private:
    //========================================
    //! \brief Construct value in place with constructor which matched with Args.
    //! \tparam Args  Type pack of constructor arguments.
    //! \param[in] args  Constructor arguments pack.
    //----------------------------------------
    template<typename... Args>
    void constructValue(Args&&... args) noexcept(std::is_nothrow_constructible<value_type, Args...>::value)
    {
        this->destructValue();

        this->m_value = new value_type{std::forward<Args>(args)...};
    }

    //========================================
    //! \brief Destruct value with default destructor.
    //----------------------------------------
    void destructValue()
    {
        if (this->m_value != nullptr)
        {
            delete this->m_value;
            this->m_value = nullptr;
        }
    }

private:
    //========================================
    //! \brief Optional value.
    //----------------------------------------
    value_type* m_value{nullptr};
}; // class Optional

//==============================================================================
//! \brief Exchange the values of two Optional container.
//! \param[in/out] lhs  Optional container to swap the values.
//! \param[in/out] rhs  Optional container to swap the values.
//------------------------------------------------------------------------------
template<typename T>
inline void swap(Optional<T>& lhs, Optional<T>& rhs) noexcept
{
    lhs.swap(rhs);
}

//==============================================================================
//! \brief Create Optional value by move constructor with
//!        in place constructed value of type T with matching constructor of argument set Args.
//! \tparam T       Any kind of value type.
//! \tparam Args    Parameter set of constructor.
//! \returns Optional value of in place constructed value of type T.
//------------------------------------------------------------------------------
template<typename T, typename... Args>
inline Optional<T> makeOptional(Args&&... args)
{
    return Optional<T>{std::move(T{std::forward<Args>(args)...})};
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
