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
//! \date Aug 29, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/io.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class Unconvertable
//! \brief A wrapper for classes to prohibit implicit type casting in generation.
//! \date Jul 21, 2011
//!
//! \tparam T  Class to be wrapped.
//------------------------------------------------------------------------------
template<class T>
class Unconvertable
{
public:
    //========================================
    //!\brief The underlying type which is wrapped by the
    //!       Unconvertable.
    //----------------------------------------
    using BaseType = T;

public:
    //========================================
    //! \brief Constructor to create an Unconvertable from an
    //!        instance of the wrapped class, the UnderlyingType. (explicit)
    //!
    //! This template class is an instrument to disable
    //! implicit compiler type casting.
    //!
    //! Casting from the class T, the UnderlyingType, to Unconvertable<T>
    //! is not allowed implicitly but only by calling the constructor
    //! explicitly.
    //!
    //! Casting from Unconvertable<T> to T is allowed implicitly
    //! by defining the cast operator.
    //!
    //! This class, e.g. can be used to define several index types
    //! (derived from integer) and you can't mix them up by accident
    //! by implicit type casting since the constructor taking an
    //! int32_t (UnderlyingType) is declared explicit.
    //!
    //! \param[in] t  Object of the to be wrapped class to setup
    //!               an Unconvertable.
    //----------------------------------------
    explicit Unconvertable(const BaseType t) : m_data(t) {}

    //========================================
    //! \brief Type cast operator to wrapped class, the UnderlyingType.
    //!
    //! Automatic cast to the wrapped class, the UnderlyingType, is allowed.
    //! The other way around (via constructor) is not
    //! allowed automatically/implicitly.
    //----------------------------------------
    operator BaseType const() const { return this->m_data; }

public:
    //========================================
    //! \brief Checks the wrapped value and an instance \a t of its underlying type for nearly equality.
    //! \param[in] t  The instance of the underlying type to be compared with
    //!               the wrapped value.
    //! \return \c true if both are nearly identical, \c false otherwise.
    //----------------------------------------
    template<typename T_ = BaseType,
             uint8_t EXP,
             typename std::enable_if<std::is_floating_point<T_>::value, int>::type = 0>
    bool fuzzyCompareT(const Unconvertable<BaseType> t) const
    {
        return microvision::common::sdk::fuzzyCompareT<EXP>(m_data, t.m_data);
    }

    //========================================
    //! \brief Checks the wrapped value and an instance \a t of its underlying type for nearly equality.
    //! \param[in] t  The instance of the underlying type to be compared with
    //!               the wrapped value.
    //! \return \c true if both are nearly identical, \c false otherwise.
    //----------------------------------------
    template<typename T_ = BaseType, typename std::enable_if<std::is_same<T_, float>::value, int>::type = 0>
    bool operator==(const Unconvertable<BaseType> t) const
    {
        return fuzzyCompareT<BaseType, 6>(t);
    }

    //========================================
    //! \brief Checks the wrapped value and an instance \a t of its underlying type for nearly equality.
    //! \param[in] t  The instance of the underlying type to be compared with
    //!               the wrapped value.
    //! \return \c true if both are nearly identical, \c false otherwise.
    //----------------------------------------
    template<typename T_ = BaseType, typename std::enable_if<std::is_same<T_, double>::value, int>::type = 0>
    bool operator==(const Unconvertable<BaseType> t) const
    {
        return fuzzyCompareT<BaseType, 14>(t);
    }

    //========================================
    //! \brief Checks the wrapped value and an instance \a t of its underlying type for equality.
    //! \param[in] t  The instance of the underlying type to be compared with
    //!               the wrapped value.
    //! \return \c True if both are identical, \c false otherwise.
    //----------------------------------------
    template<typename T_ = BaseType, typename std::enable_if<!std::is_floating_point<T_>::value, int>::type = 0>
    bool operator==(const Unconvertable<BaseType> t) const
    {
        return m_data == t.m_data;
    }

    //========================================
    //! \brief Checks the wrapped value and an instance \a t of its underlying type for nearly inequality.
    //! \param[in] t  The instance of the underlying type to be compared with
    //!               the wrapped value.
    //! \return \c true if both are not nearly identical, \c false otherwise.
    //----------------------------------------
    template<typename T_ = BaseType, typename std::enable_if<std::is_floating_point<T_>::value, int>::type = 0>
    bool operator!=(const Unconvertable<BaseType> t) const
    {
        return !(*this == t);
    }

    //========================================
    //! \brief Checks the wrapped value and an instance \a t of its underlying type for inequality.
    //! \param[in] t  The instance of the underlying type to be compared with
    //!               the wrapped value.
    //! \return \c True if both are not identical, \c false otherwise.
    //----------------------------------------
    template<typename T_ = T, typename std::enable_if<!std::is_floating_point<T_>::value, int>::type = 0>
    bool operator!=(const Unconvertable<BaseType> t) const
    {
        return !(*this == t);
    }

protected:
    //========================================
    //! \brief Wrapped object.
    //----------------------------------------
    BaseType m_data;
}; // Unconvertable

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \class ComparableUnconvertable
//! \brief A wrapper for classes to prohibit implicit type casting in generation.
//!    For types that have a lesser than operator. (explicit)
//! \date Jul 21, 2011
//!
//! \tparam T  Class to be wrapped.
//------------------------------------------------------------------------------
template<class T>
class ComparableUnconvertable
{
public:
    //========================================
    //!\brief The underlying type which is wrapped by the
    //!       ComparableUnconvertable.
    //----------------------------------------
    using BaseType = T;

public:
    //========================================
    //! \brief Constructor to create an ComparableUnconvertable from an
    //!        instance of the wrapped class.
    //!
    //! This template class is an instrument to disable
    //! implicit compiler type casting.
    //!
    //! Casting from the class T, the UnderlyingType, to ComparableUnconvertable<T>
    //! is not allowed implicitly but only by calling the constructor
    //! explicitly.
    //!
    //! Casting from ComparableUnconvertable<T> to its UnderlyingType is allowed
    //! implicitly by defining the cast operator.
    //!
    //! The only difference to class Unconvertable is that there is
    //! a lesser than and a greater than operator available.
    //!
    //! This class, e.g. can be used to define several index types
    //! (derived from integer) and you can't mix them up by accident
    //! by implicit type casting since the constructor taking an
    //! int32_t (UnderlyingType) is declared explicit.
    //!
    //! \param[in] t  Object of the to be wrapped class to setup
    //!               an ComparableUnconvertable.
    //----------------------------------------
    explicit ComparableUnconvertable(const BaseType t) : m_data(t) {}

    //========================================
    //! \brief Type cast operator to wrapped class.
    //!
    //! Automatic cast to the wrapped class is allowed.
    //! The other way around (via constructor) is not
    //! allowed automatically/implicitly.
    //----------------------------------------
    operator BaseType const() const { return this->m_data; }

    //========================================
    //! \brief Lesser than operator for ComparableUnconvertable.
    //!
    //! \param[in] l  Left hand operand of <.
    //! \param[in] r  Light hand operand of <.
    //!
    //! \return \c True if \a l<\a r, \c false otherwise.
    //----------------------------------------
    friend bool operator<(const ComparableUnconvertable l, const ComparableUnconvertable r)
    {
        return l.m_data < r.m_data;
    }

    //========================================
    //! \brief Lesser than operator for ComparableUnconvertable and its UnderlyingType.
    //!
    //! \param[in] l  Left hand operand of <.
    //! \param[in] r  Light hand operand of <.
    //!
    //! \return \c True if \a l<\a r, \c false otherwise.
    //----------------------------------------
    friend bool operator<(const ComparableUnconvertable l, const BaseType r) { return l.m_data < r; }

    //========================================
    //! \brief Lesser than operator for ComparableUnconvertable and its UnderlyingType.
    //!
    //! \param[in] l  Left hand operand of <.
    //! \param[in] r  Light hand operand of <.
    //!
    //! \return \c True if \a l<\a r, \c false otherwise.
    //----------------------------------------
    friend bool operator<(const BaseType l, const ComparableUnconvertable r) { return l < r.m_data; }

    //========================================
    //! \brief Greater than operator for ComparableUnconvertable.
    //!
    //! \param[in] l  Left hand operand of >.
    //! \param[in] r  Light hand operand of >.
    //!
    //! \return \c True if \a l>\a r, \c false otherwise.
    //----------------------------------------
    friend bool operator>(const ComparableUnconvertable l, const ComparableUnconvertable r)
    {
        return l.m_data > r.m_data;
    }

    //========================================
    //! \brief Greater than operator for ComparableUnconvertable and its UnderlyingType.
    //!
    //! \param[in] l  Left hand operand of >.
    //! \param[in] r  Light hand operand of >.
    //!
    //! \return \c True if \a l>\a r, \c false otherwise.
    //----------------------------------------
    friend bool operator>(const ComparableUnconvertable l, const BaseType r) { return l.m_data > r; }

    //========================================
    //! \brief Greater than operator for ComparableUnconvertable and its UnderlyingType.
    //!
    //! \param[in] l  Left hand operand of >.
    //! \param[in] r  Light hand operand of >.
    //!
    //! \return \c True if \a l>\a r, \c false otherwise.
    //----------------------------------------
    friend bool operator>(const BaseType l, const ComparableUnconvertable r) { return l > r.m_data; }

protected:
    //========================================
    //! \brief Wrapped object.
    //----------------------------------------
    BaseType m_data;
}; // ComparableUnconvertable

//==============================================================================

//==============================================================================
//! \brief Id of an object.
//! \date Feb 22, 2016
//!
//------------------------------------------------------------------------------
class ObjectId final : public ComparableUnconvertable<uint16_t>
{
public:
    //========================================
    //! \brief Construtor of ObjectId.
    //!
    //! \param[in] oid  Object Id as integer.
    //----------------------------------------
    explicit ObjectId(const uint16_t oid) : ComparableUnconvertable<uint16_t>(oid) {}
    ObjectId() : ComparableUnconvertable<uint16_t>(0) {}

    bool isset() const { return (this->m_data > 0 && this->m_data != 65535); }
    void unset() { this->m_data = 0; }

public:
    static std::streamsize getSerializedSize() { return sizeof(uint16_t); }

public:
    std::istream& read(std::istream& is)
    {
        uint16_t tmp;
        microvision::common::sdk::read(is, tmp);
        this->m_data = tmp;
        return is;
    }

    std::ostream& write(std::ostream& os) const
    {
        microvision::common::sdk::write(os, uint16_t(this->m_data));
        return os;
    }
}; // ObjectId

//==============================================================================

//==============================================================================
//! \brief Id of an object.
//! \date Feb 22, 2016
//!
//------------------------------------------------------------------------------
class ObjectId32 final : public ComparableUnconvertable<uint32_t>
{
public:
    //========================================
    //! \brief Constructor of ObjectId.
    //!
    //! \param[in] oid  Object Id as integer.
    //----------------------------------------
    explicit ObjectId32(const uint32_t oid) : ComparableUnconvertable<uint32_t>(oid) {}
    ObjectId32(const ObjectId oid) : ComparableUnconvertable<uint32_t>(oid)
    {
        if (!oid.isset())
            this->unset();
    }
    ObjectId32() : ComparableUnconvertable<uint32_t>(0) {}

    bool isset() const { return (this->m_data > 0 && this->m_data != ((2L ^ 32) - 1)); }
    void unset() { this->m_data = 0; }

public:
    static std::streamsize getSerializedSize() { return sizeof(uint32_t); }

public:
    std::istream& read(std::istream& is)
    {
        uint32_t tmp;
        microvision::common::sdk::read(is, tmp);
        this->m_data = tmp;
        return is;
    }

    std::ostream& write(std::ostream& os) const
    {
        microvision::common::sdk::write(os, uint32_t(this->m_data));
        return os;
    }
}; // ObjectId32

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
