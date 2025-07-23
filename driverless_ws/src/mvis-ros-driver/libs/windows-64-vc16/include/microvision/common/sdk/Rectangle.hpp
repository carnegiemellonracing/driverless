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
//! \date Jan 11, 2018
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/io.hpp>

#include <iostream>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class Rectangle
//! \brief Vector class for which can store a rectangle with center point and
//!        size (x and y extension) and a rotation.
// ------------------------------------------------------------------------------
template<typename T>
class Rectangle final
{
public: // type definitions
    //========================================
    using ValueType = T;

public: // constructors
    //========================================
    //! \brief Default constructor.
    //!
    //! This constructor initialize all values to be 0.
    //----------------------------------------
    Rectangle() : m_center{}, m_size{}, m_rotation{0} {}

    //========================================
    //! \brief Constructor.
    //! \param[in] center    The center for the rectangle.
    //! \param[in] size      The extension in x and y direction.
    //! \param[in] rotation  The rotation of the rectangle.
    //----------------------------------------
    Rectangle(const Vector2<ValueType>& center, const Vector2<ValueType>& size, const ValueType rotation = 0)
      : m_center{center}, m_size{size}, m_rotation{rotation}
    {}

public: // operators
    //========================================
    //! \brief Operator for comparing this rectangle with another for equality.
    //! \tparam T  A non-floating point type.
    //! \param[in] other  The rectangle which shall be compared with this one.
    //! \return \c true, if center, size and rotation are equal,
    //!         \c false otherwise.
    //----------------------------------------
    template<typename T_ = T, typename std::enable_if<!std::is_floating_point<T_>::value, int>::type = 0>
    bool operator==(const Rectangle<ValueType>& other) const
    {
        return (m_center == other.m_center) //
               && (m_size == other.m_size) && //
               (m_rotation == other.m_rotation);
    }

    //========================================
    //! \brief Operator for comparing this rectangle with another one to be about the same. NaN equals NaN here.
    //! \tparam T  float
    //! \param[in] other  The rectangle which shall be compared with this one.
    //! \return \c true, if center, size and rotation are about the same,
    //!         \c false otherwise.
    //! \note In this method two double numbers will be assumed to be equal, if the absolute value
    //!       of the difference between them is less than 1e-12.
    //----------------------------------------
    template<typename T_ = T, typename std::enable_if<std::is_same<T_, float>::value, int>::type = 0>
    bool operator==(const Rectangle<ValueType>& other) const
    {
        return (m_center == other.m_center) //
               && (m_size == other.m_size) //
               && fuzzyFloatEqualT<12>(m_rotation, other.m_rotation);
    }

    //========================================
    //! \brief Operator for comparing this rectangle with another one to be about the same. NaN equals NaN here.
    //! \tparam T  double
    //! \param[in] other  The rectangle which shall be compared with this one.
    //! \return \c true, if center, size and rotation are about the same,
    //!         \c false otherwise.
    //! \note In this method two double numbers will be assumed to be equal, if the absolute value
    //!       of the difference between them is less than 1e-17.
    //----------------------------------------
    template<typename T_ = T, typename std::enable_if<std::is_same<T_, double>::value, int>::type = 0>
    bool operator==(const Rectangle<ValueType>& other) const
    {
        return (m_center == other.m_center) //
               && (m_size == other.m_size) //
               && fuzzyDoubleEqualT<17>(m_rotation, other.m_rotation);
    }

    //========================================
    //! \brief Comparing this rectangle and another one to be about the same. NaN equals NaN here.
    //! \param[in] other  The rectangle which shall be compared with this one.
    //! \return \c true, if center and size are equal and rotation is nearly equal,
    //!         \c false otherwise
    //----------------------------------------
    template<typename T_ = T, uint8_t EXP, typename std::enable_if<std::is_floating_point<T_>::value, int>::type = 0>
    bool fuzzyCompareT(const Rectangle<ValueType>& other) const
    {
        return fuzzyCompareT<EXP>(m_center, other.m_center) //
               && fuzzyCompareT<EXP>(m_size, other.m_size) //
               && fuzzyCompareT<EXP>(m_rotation, other.m_rotation);
    }

    //========================================
    //! \brief Operator comparing two rectangles for inequality. NaN equals NaN here.
    //! \param[in] other  The rectangle which shall be compared to this one.
    //! \return \c false, if the rectangles are not about to be the same,
    //!         \c true otherwise.
    //! \sa Rectangle::operator==.
    //----------------------------------------
    bool operator!=(const Rectangle<ValueType>& other) const { return !(*this == other); }

public: // getters and setters
    //========================================
    //! \brief Gets for the center point of the rectangle.
    //! \return The center point of the rectangle.
    //----------------------------------------
    const Vector2<T>& getCenter() const { return m_center; }

    //========================================
    //! \brief Sets the center point of the rectangle.
    //! \param[in] center  The new center point for the rectangle.
    //----------------------------------------
    void setCenter(const Vector2<T>& center) { m_center = center; }

    //========================================
    //! \brief Gets for the size of the rectangle.
    //! \return The size of the rectangle.
    //----------------------------------------
    const Vector2<T>& getSize() const { return m_size; }

    //========================================
    //! \brief Sets the size of the rectangle.
    //! \param[in] size  The new size of the rectangle.
    //----------------------------------------
    void setSize(const Vector2<T>& size) { m_size = size; }

    //========================================
    //! \brief Gets for the rotation of the rectangle.
    //! \return The rotation angle in rad.
    //----------------------------------------
    const T getRotation() const { return m_rotation; }

    //========================================
    //! \brief Sets the rotation of the rectangle.
    //! \param[in] angle  The rotation angle in rad.
    //----------------------------------------
    void setRotation(const T angle) { m_rotation = angle; }

public: // member functions
    //========================================
    //! \brief Scales the rectangle by the given \a factor.
    //! \param[in] factor  The factor used for scaling.
    //! \return A reference to this after scaling.
    //----------------------------------------
    Rectangle<ValueType>& scale(const ValueType factor)
    {
        m_size.scale(factor);
        return *this;
    }

    //========================================
    //! \brief Scales the rectangle by the given \a factor.
    //! \param[in] factor  The factor used for scaling.
    //! \return A scaled version of this.
    //----------------------------------------
    Rectangle<ValueType> getScaled(const ValueType factor) const
    {
        Rectangle<ValueType> ret{*this};
        return ret.scale(factor);
    }

    //========================================
    //! \brief Translates the center of this rectangle by a given \a offset.
    //! \param[in] offset  The offset about the rectangle shall be translated.
    //! \return A reference to this after translation.
    //----------------------------------------
    Rectangle<ValueType>& translate(const Vector2<ValueType>& offset)
    {
        m_center += offset;
        return *this;
    }

    //========================================
    //! \brief Translates the center of this rectangle by a given \a offset.
    //! \param[in] offset  The offset about the rectangle will be translated.
    //! \return A translated version of this.
    //----------------------------------------
    Rectangle<ValueType> getTranslated(const Vector2<ValueType>& offset) const
    {
        Rectangle<ValueType> ret{*this};
        return ret.translate(offset);
    }

    //========================================
    //! \brief Rotates the rectangle about a given \a angle.
    //! \param[in] angle  The angle in rad about the rectangle will be rotated.
    //! \return A reference to this after rotation.
    //----------------------------------------
    Rectangle<ValueType>& rotate(const ValueType angle)
    {
        m_rotation += angle;
        return *this;
    }

    //========================================
    //! \brief Rotates the rectangle about a given \a angle.
    //! \param[in] angle  The angle in rad about the rectangle will be rotated.
    //! \return A rotated version of this.
    //----------------------------------------
    Rectangle<ValueType> getRotated(const ValueType angle) const
    {
        Rectangle<ValueType> ret{*this};
        return ret.rotate(angle);
    }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, Rectangle<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const Rectangle<TT>& value);

public:
    static constexpr bool isSerializable()
    {
        return (std::is_same<ValueType, float>{} || std::is_same<ValueType, int16_t>{});
    }

protected: // members
    Vector2<T> m_center;
    Vector2<T> m_size;
    T m_rotation;

}; // Rectangle

//==============================================================================
// Serialization
//==============================================================================

//==============================================================================
template<typename T>
inline constexpr std::streamsize serializedSize(const Rectangle<T>&)
{
    static_assert(Rectangle<T>::isSerializable(),
                  "serializedSize is not implemented for given template type of Rectangle");

    return 2 * serializedSize(Vector2<T>{}) + serializedSize(T{});
}

//==============================================================================
template<typename T>
inline void writeBE(std::ostream& os, const Rectangle<T>& p)
{
    static_assert(Rectangle<T>::isSerializable(), "writeBE is not implemented for given template type of Rectangle");

    microvision::common::sdk::writeBE(os, p.m_center);
    microvision::common::sdk::writeBE(os, p.m_size);
    microvision::common::sdk::writeBE(os, p.m_rotation);
}

//==============================================================================
template<typename T>
inline void readBE(std::istream& is, Rectangle<T>& p)
{
    static_assert(Rectangle<T>::isSerializable(), "readBE is not implemented for given template type of Rectangle");

    microvision::common::sdk::readBE(is, p.m_center);
    microvision::common::sdk::readBE(is, p.m_size);
    microvision::common::sdk::readBE(is, p.m_rotation);
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision
