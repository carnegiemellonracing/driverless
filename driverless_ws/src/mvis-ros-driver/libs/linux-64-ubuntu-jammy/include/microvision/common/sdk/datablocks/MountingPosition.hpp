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
//! \date Jan 16, 2018
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/io.hpp> //_prototypes.hpp>

#include <iostream>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\class MountingPosition
//!\brief MountingPosition class which stores a mounting position of a device.
//------------------------------------------------------------------------------
template<typename T>
class MountingPosition final
{
public: // type definitions
    //========================================
    using ValueType = T;

public:
    static constexpr bool isSerializable()
    {
        return (std::is_same<ValueType, float>{} || std::is_same<ValueType, int16_t>{});
    }

public: // constructors
    //========================================
    //! \brief Default constructor, initializes all angles and positions to 0
    //----------------------------------------
    MountingPosition() : m_rotation{}, m_position{} {}

    //========================================
    //! \brief Constructor with member initialization
    //! \param[in] yaw yaw angle
    //! \param[in] pitch pitch angle
    //! \param[in] roll roll angle
    //! \param[in] x x position
    //! \param[in] y y position
    //! \param[in] z z position
    //----------------------------------------
    MountingPosition(const ValueType yaw,
                     const ValueType pitch,
                     const ValueType roll,
                     const ValueType x,
                     const ValueType y,
                     const ValueType z)
      : m_rotation{roll, pitch, yaw}, m_position{x, y, z}
    {}

    //========================================
    //! \brief Constructor with member initialization
    //! \param[in] rotation The rotation vector with order roll, pitch, yaw
    //! (x-, y-, z-axis)
    //! \param[in] position The position vector
    //----------------------------------------
    MountingPosition(const Vector3<ValueType>& rotation, const Vector3<ValueType>& position)
      : m_rotation{rotation}, m_position{position}
    {}

    //========================================
    //! \brief Default Destructor
    //----------------------------------------
    virtual ~MountingPosition() {}

public: // member functions
    //========================================
    //! \brief Getter function for the yaw angle
    //! \return The yaw angle of the mounting position
    //----------------------------------------
    ValueType getYaw() const { return m_rotation.getZ(); }

    //========================================
    //! \brief Getter function for the pitch angle
    //! \return The pitch angle of the mounting position
    ValueType getPitch() const { return m_rotation.getY(); }

    //========================================
    //! \brief Getter function for the roll angle
    //! \return The roll angle of the mounting position
    ValueType getRoll() const { return m_rotation.getX(); }

    //========================================
    //! \brief Getter function for the x value
    //! \return The x component of the vector
    //----------------------------------------
    ValueType getX() const { return m_position.getX(); }

    //========================================
    //! \brief Getter function for the y value
    //! \return The y component of the vector
    //----------------------------------------
    ValueType getY() const { return m_position.getY(); }

    //========================================
    //! \brief Getter function for the z value
    //! \return The z component of the vector
    //----------------------------------------
    ValueType getZ() const { return m_position.getZ(); }

    //========================================
    //! \brief Getter for the position vector
    //! \return A constant reference to the position vector
    //----------------------------------------
    const Vector3<ValueType>& getPosition() const { return m_position; }

    //========================================
    //! \brief Getter for the rotation vector (roll, pitch, yaw)
    //! \return A constant reference to the rotation vector
    //----------------------------------------
    const Vector3<ValueType>& getRotation() const { return m_rotation; }

    //========================================
    //! \brief Setter function for the yaw angle
    //! \param[in] val The value which will replace the current yaw angle
    //----------------------------------------
    void setYaw(const ValueType val) { m_rotation.setZ(val); }

    //========================================
    //! \brief Setter function for the pitch angle
    //! \param[in] val The value which will replace the current pitch angle
    //----------------------------------------
    void setPitch(const ValueType val) { m_rotation.setY(val); }

    //========================================
    //! \brief Setter function for the roll angle
    //! \param[in] val The value which will replace the current roll angle
    //----------------------------------------
    void setRoll(const ValueType val) { m_rotation.setX(val); }

    //========================================
    //! \brief Setter function for the x value
    //! \param[in] val The value which will replace the current x value
    //----------------------------------------
    void setX(const ValueType val) { m_position.setX(val); }

    //========================================
    //! \brief Setter function for the y value
    //! \param[in] val The value which will replace the current y value
    //----------------------------------------
    void setY(const ValueType val) { m_position.setY(val); }

    //========================================
    //! \brief Setter function for the z value
    //! \param[in] val The value which will replace the current y value
    //----------------------------------------
    void setZ(const ValueType val) { m_position.setZ(val); }

    //========================================
    //! \brief Setter for the position vector
    //! \param[in] position The new position vector
    //----------------------------------------
    void setPosition(const Vector3<ValueType>& position) { m_position = position; }

    //========================================
    //! \brief Setter for the position vector (roll, pitch, yaw)
    //! \param[in] rotation The new position vector
    //----------------------------------------
    void setRotation(const Vector3<ValueType>& rotation) { m_rotation = rotation; }

public: // member functions
    //========================================
    //! \brief Normalizes the angles to be in the range of [-PI,PI]
    //! \return A reference to this after normalization
    //----------------------------------------
    MountingPosition<ValueType>& normalizeAngles()
    {
        m_rotation.setX(microvision::common::sdk::normalizeRadians(m_rotation.getX()));
        m_rotation.setY(microvision::common::sdk::normalizeRadians(m_rotation.getY()));
        m_rotation.setZ(microvision::common::sdk::normalizeRadians(m_rotation.getZ()));
        return *this;
    }

    //========================================

protected: // member variables
    Vector3<ValueType> m_rotation;
    Vector3<ValueType> m_position;

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, MountingPosition<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const MountingPosition<TT>& value);

    template<typename TT>
    friend std::ostream& operator<<(std::ostream& is, const MountingPosition<TT>& mountingPosition);
    template<typename TT>
    friend std::istream& operator>>(std::istream& is, MountingPosition<TT>& mountingPosition);
    template<typename TT>
    friend void readLE(std::istream& is, MountingPosition<TT>& value);
    template<typename TT>
    friend void writeLE(std::ostream& os, const MountingPosition<TT>& value);
}; // MountingPosition

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Operator for comparing two mounting positions for equality.
//! \param[in] lhs  The first mounting position to compare.
//! \param[in] rhs  The second mounting position to compare.
//! \return True, if all angles and positions are equal, false otherwise.
//------------------------------------------------------------------------------
template<typename T>
inline bool operator==(const MountingPosition<T>& lhs, const MountingPosition<T>& rhs)
{
    return (lhs.getRotation() == rhs.getRotation()) //
           && (lhs.getPosition() == rhs.getPosition());
}

//==============================================================================
//! \brief Operator for comparing two mounting positions for inequality
//! \param[in] lhs  The first mounting position to compare
//! \param[in] rhs  The second mounting position to compare
//! \return True, if any angle or positions is inequal, false otherwise
//------------------------------------------------------------------------------
template<typename T>
inline bool operator!=(const MountingPosition<T>& lhs, const MountingPosition<T>& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
//! \brief Tests whether two mounting positions are nearly equal.
//! \tparam EXP     The exponent of the epsilon used for the fuzzy
//!                 compare. 10^(-EXP).
//! \param[in] lhs  First value to be compared with second value.
//! \param[in] rhs  Second value to be compared with first value.
//! \return \c True if the two \c mounting positions are equal in
//!         terms of the machine precision, which means their
//!         difference must be less than 10^(-EXP).
//!
//! The exponent range is defined in NegFloatPotenciesOf10.
//------------------------------------------------------------------------------
template<typename T, uint8_t EXP>
inline bool fuzzyCompareT(const MountingPosition<T>& lhs, const MountingPosition<T>& rhs)
{
    return fuzzyCompareT<EXP>(lhs.getPosition(), rhs.getPosition()) //
           && fuzzyCompareT<EXP>(lhs.getRotation(), rhs.getRotation());
}

//==============================================================================
//! \brief Fuzzy Compare two mounting positions \a a and \a b. NaN is equal NaN here.
//! \tparam EXP     The exponent of the epsilon used for the fuzzy
//!                 compare. 10^(-EXP).
//! \param[in] lhs  First vector to be compared.
//! \param[in] rhs  Second vector to be compared.
//! \return \c True if the difference between \a a and \a b is not smaller
//!         than 10^(-EXP) or if both are NaN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<typename T, uint8_t EXP>
inline bool fuzzyEqualT(const MountingPosition<T>& lhs, const MountingPosition<T>& rhs)
{
    return fuzzyEqualT<EXP>(lhs.getPosition(), rhs.getPosition()) //
           && fuzzyEqualT<EXP>(lhs.getRotation(), rhs.getRotation());
}

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Stream operator for writing the mounting position to a stream
//! \param[in, out] os                The stream, the mounting position shall be written to
//! \param[in]      mountingPosition  The mounting position which shall be streamed
//! \return The stream to which the mounting position was written to
//------------------------------------------------------------------------------
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const MountingPosition<T>& mountingPosition)
{
    os << "(" << mountingPosition.m_position << "," << mountingPosition.m_rotation << ")";
    return os;
}

//==============================================================================
//! \brief Stream operator for reading the mounting position from a stream
//! \param[in, out] is                The stream, the mounting position shall be read from
//! \param[out]     mountingPosition  The mounting position as read from the stream
//! \return The stream from which the mounting position was read
//!
//! \note If reading the data failed (check with /a istream::fail()) the content of the mountingPosition is undefined.
//------------------------------------------------------------------------------
template<typename T>
inline std::istream& operator>>(std::istream& is, MountingPosition<T>& mountingPosition)
{
    char charBuffer;

    is.get(charBuffer);
    if (charBuffer != '(')
    {
        // Wrong prefix.
        is.setstate(std::ios::failbit);
        return is;
    }

    is >> mountingPosition.m_position;

    is.get(charBuffer);
    if (charBuffer != ',')
    {
        // Wrong separator.
        is.setstate(std::ios::failbit);
        return is;
    }

    is >> mountingPosition.m_rotation;

    is.get(charBuffer);
    if (charBuffer != ')')
    {
        // Wrong suffix.
        is.setstate(std::ios::failbit);
        return is;
    }

    return is;
}

//==============================================================================

//==============================================================================
// Serialization
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const MountingPosition<T>& p)
{
    static_assert(MountingPosition<T>::isSerializable(),
                  "writeBE is not implemented for given template type of MountingPosition");

    microvision::common::sdk::writeBE(os, p.getYaw());
    microvision::common::sdk::writeBE(os, p.getPitch());
    microvision::common::sdk::writeBE(os, p.getRoll());
    microvision::common::sdk::writeBE(os, p.m_position);
}

//==============================================================================

template<typename T>
inline void readBE(std::istream& is, MountingPosition<T>& p)
{
    static_assert(MountingPosition<T>::isSerializable(),
                  "readBE is not implemented for given template type of MountingPosition");

    T yaw, pitch, roll;
    microvision::common::sdk::readBE(is, yaw);
    microvision::common::sdk::readBE(is, pitch);
    microvision::common::sdk::readBE(is, roll);
    microvision::common::sdk::readBE(is, p.m_position);

    p.setYaw(yaw);
    p.setPitch(pitch);
    p.setRoll(roll);
}

//==============================================================================

template<typename T>
inline void writeLE(std::ostream& os, const MountingPosition<T>& p)
{
    static_assert(MountingPosition<T>::isSerializable(),
                  "writeLE is not implemented for given template type of MountingPosition");

    microvision::common::sdk::writeLE(os, p.getYaw());
    microvision::common::sdk::writeLE(os, p.getPitch());
    microvision::common::sdk::writeLE(os, p.getRoll());
    microvision::common::sdk::writeLE(os, p.m_position);
}

template<typename T>
inline void readLE(std::istream& is, MountingPosition<T>& p)
{
    static_assert(MountingPosition<T>::isSerializable(),
                  "readLE is not implemented for given template type of MountingPosition");

    T yaw, pitch, roll;
    microvision::common::sdk::readLE(is, yaw);
    microvision::common::sdk::readLE(is, pitch);
    microvision::common::sdk::readLE(is, roll);
    microvision::common::sdk::readLE(is, p.m_position);

    p.setYaw(yaw);
    p.setPitch(pitch);
    p.setRoll(roll);
}

//==============================================================================

template<typename T>
inline constexpr std::streamsize serializedSize(const MountingPosition<T>&)
{
    static_assert(MountingPosition<T>::isSerializable(),
                  "serializedSize is not implemented for given template type of MountingPosition");

    return std::streamsize{3 * sizeof(T)} + serializedSize(Vector3<T>{});
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
