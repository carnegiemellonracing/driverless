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
#include <microvision/common/sdk/io.hpp>

#include <iostream>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief This class stores a mounting position of a device with the standard deviation.
//------------------------------------------------------------------------------
template<typename T>
class MountingPositionWithDeviation final
{
public: // type definitions
    using ValueType = T;

public:
    static constexpr bool isSerializable()
    {
        return (std::is_same<ValueType, float>{} || std::is_same<ValueType, int16_t>{});
    }

public: // constructors
    //========================================
    //! \brief Default constructor.
    //! Initializes all angles and positions to 0.
    //----------------------------------------
    MountingPositionWithDeviation() : m_position{}, m_rotation{}, m_positionDev{}, m_rotationDev{} {}

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] x       The x position.
    //! \param[in] y       The y position.
    //! \param[in] z       The z position.
    //! \param[in] yaw     The yaw angle.
    //! \param[in] pitch   The pitch angle.
    //! \param[in] roll    The roll angle.
    //! \param[in] xd      The x position standard deviation.
    //! \param[in] yd      The y position standard deviation.
    //! \param[in] zd      The z position standard deviation.
    //! \param[in] yawd    The yaw angle standard deviation.
    //! \param[in] pitchd  The pitch angle standard deviation.
    //! \param[in] rolld   The roll angle standard deviation.
    //----------------------------------------
    MountingPositionWithDeviation(const ValueType x,
                                  const ValueType y,
                                  const ValueType z,
                                  const ValueType yaw,
                                  const ValueType pitch,
                                  const ValueType roll,
                                  const ValueType xd,
                                  const ValueType yd,
                                  const ValueType zd,
                                  const ValueType yawd,
                                  const ValueType pitchd,
                                  const ValueType rolld)
      : m_position{x, y, z}, m_rotation{roll, pitch, yaw}, m_positionDev{xd, yd, zd}, m_rotationDev{rolld, pitchd, yawd}
    {}

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] position     The position vector (x-, y-, z-axis).
    //! \param[in] rotation     The rotation vector with order roll, pitch, yaw.
    //! \param[in] positionDev  The position standard deviation vector.
    //! \param[in] rotationDev  The rotation standard deviation vector vector.
    //----------------------------------------
    MountingPositionWithDeviation(const Vector3<ValueType>& position,
                                  const Vector3<ValueType>& rotation,
                                  const Vector3<ValueType>& positionDev,
                                  const Vector3<ValueType>& rotationDev)
      : m_position{position}, m_rotation{rotation}, m_positionDev{positionDev}, m_rotationDev{rotationDev}
    {}

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    virtual ~MountingPositionWithDeviation() {}

public: //getter
    //========================================
    //! \brief Get the x value in m.
    //! \return The x component of the vector.
    //----------------------------------------
    ValueType getX() const { return m_position.getX(); }

    //========================================
    //! \brief Get the y value in m.
    //! \return The y component of the vector.
    //----------------------------------------
    ValueType getY() const { return m_position.getY(); }

    //========================================
    //! \brief Get the z value in m.
    //! \return The z component of the vector.
    //----------------------------------------
    ValueType getZ() const { return m_position.getZ(); }

    //========================================
    //! \brief Get the yaw angle in rad.
    //! \return The yaw angle of the mounting position.
    //----------------------------------------
    ValueType getYaw() const { return m_rotation.getZ(); }

    //========================================
    //! \brief Get the pitch angle in rad.
    //! \return The pitch angle of the mounting position.
    ValueType getPitch() const { return m_rotation.getY(); }

    //========================================
    //! \brief Get the roll angle in rad.
    //! \return The roll angle of the mounting position.
    ValueType getRoll() const { return m_rotation.getX(); }

    //========================================
    //! \brief Get the position vector in m.
    //! \return A constant reference to the position vector.
    //----------------------------------------
    const Vector3<ValueType>& getPosition() const { return m_position; }

    //========================================
    //! \brief Get the rotation vector (roll, pitch, yaw) in rad.
    //! \return A constant reference to the rotation vector.
    //----------------------------------------
    const Vector3<ValueType>& getRotation() const { return m_rotation; }

    //========================================
    //! \brief Get the position standard deviation vector in m.
    //! \return A constant reference to the position standard deviation vector.
    //----------------------------------------
    const Vector3<ValueType>& getPositionStandardDeviation() const { return m_positionDev; }

    //========================================
    //! \brief Get the rotation standard deviation vector (roll, pitch, yaw) in rad.
    //! \return A constant reference to the rotation standard deviation vector.
    //----------------------------------------
    const Vector3<ValueType>& getRotationStandardDeviation() const { return m_rotationDev; }

public: // setter
    //========================================
    //! \brief Set the x value in m.
    //! \param[in] val  The value which will replace the current x value.
    //----------------------------------------
    void setX(const ValueType val) { m_position.setX(val); }

    //========================================
    //! \brief Set the y value in m.
    //! \param[in] val  The value which will replace the current y value.
    //----------------------------------------
    void setY(const ValueType val) { m_position.setY(val); }

    //========================================
    //! \brief Set the z value in m.
    //! \param[in] val  The value which will replace the current y value.
    //----------------------------------------
    void setZ(const ValueType val) { m_position.setZ(val); }

    //========================================
    //! \brief Set the yaw angle in rad.
    //! \param[in] val  The value which will replace the current yaw angle.
    //----------------------------------------
    void setYaw(const ValueType val) { m_rotation.setZ(val); }

    //========================================
    //! \brief Set the pitch angle in rad.
    //! \param[in] val  The value which will replace the current pitch angle.
    //----------------------------------------
    void setPitch(const ValueType val) { m_rotation.setY(val); }

    //========================================
    //! \brief Set the roll angle in rad.
    //! \param[in] val  The value which will replace the current roll angle.
    //----------------------------------------
    void setRoll(const ValueType val) { m_rotation.setX(val); }

    //========================================
    //! \brief Set the position vector in m.
    //! \param[in] position  The new position vector.
    //----------------------------------------
    void setPosition(const Vector3<ValueType>& position) { m_position = position; }

    //========================================
    //! \brief Set the position vector (roll, pitch, yaw)in rad.
    //! \param[in] rotation  The new position vector.
    //----------------------------------------
    void setRotation(const Vector3<ValueType>& rotation) { m_rotation = rotation; }

    //========================================
    //! \brief Set the position standard deviation vector in m.
    //! \param[in] position  The new position standard deviation vector.
    //----------------------------------------
    void setPositionStandardDeviation(const Vector3<ValueType>& position) { m_positionDev = position; }

    //========================================
    //! \brief Set the position standard deviation vector (roll, pitch, yaw) in rad.
    //! \param[in] rotation  The new position standard deviation vector.
    //----------------------------------------
    void setRotationStandardDeviation(const Vector3<ValueType>& rotation) { m_rotationDev = rotation; }

protected: // member variables
    Vector3<ValueType> m_position{}; //!< The position in m.
    Vector3<ValueType> m_rotation{}; //!< The rotation in rad.
    Vector3<ValueType> m_positionDev{}; //!< The standard deviation of the position in m.
    Vector3<ValueType> m_rotationDev{}; //!< The standard deviation of the rotation in rad.

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, MountingPositionWithDeviation<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const MountingPositionWithDeviation<TT>& value);

    template<typename TT>
    friend std::ostream& operator<<(std::ostream& is, const MountingPositionWithDeviation<TT>& mountingPosition);
    template<typename TT>
    friend std::istream& operator>>(std::istream& is, MountingPositionWithDeviation<TT>& mountingPosition);
    template<typename TT>
    friend void readLE(std::istream& is, MountingPositionWithDeviation<TT>& value);
    template<typename TT>
    friend void writeLE(std::ostream& os, const MountingPositionWithDeviation<TT>& value);
}; // MountingPositionWithDeviation

//==============================================================================
// Specializations for operators.
//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
template<typename T>
inline bool operator==(const MountingPositionWithDeviation<T>& lhs, const MountingPositionWithDeviation<T>& rhs)
{
    return (lhs.getPosition() == rhs.getPosition()) //
           && (lhs.getRotation() == rhs.getRotation()) //
           && (lhs.getPositionStandardDeviation() == rhs.getPositionStandardDeviation()) //
           && (lhs.getRotationStandardDeviation() == rhs.getRotationStandardDeviation());
}

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
template<typename T>
inline bool operator!=(const MountingPositionWithDeviation<T>& lhs, const MountingPositionWithDeviation<T>& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
//! \brief Tests whether two mounting positions are nearly equal.
//! \tparam T       The value type of the mounting position.
//! \tparam EXP     The exponent of the epsilon used for the fuzzy compare. 10^(-EXP).
//! \param[in] lhs  First value to be compared with second value.
//! \param[in] rhs  Second value to be compared with first value.
//! \return \c True if the two \c mounting positions are equal in terms of the machine precision,
//!         which means their difference must be less than 10^(-EXP).
//!         \c false otherwise.
//!
//! The exponent range is defined in NegFloatPotenciesOf10.
//------------------------------------------------------------------------------
template<typename T, uint8_t EXP>
inline bool fuzzyCompareT(const MountingPositionWithDeviation<T>& lhs, const MountingPositionWithDeviation<T>& rhs)
{
    return fuzzyCompareT<EXP>(lhs.getPosition(), rhs.getPosition()) //
           && fuzzyCompareT<EXP>(lhs.getRotation(), rhs.getRotation()) //
           && fuzzyCompareT<EXP>(lhs.getPositionStandardDeviation(), rhs.getPPositionStandardDeviation()) //
           && fuzzyCompareT<EXP>(lhs.getRotationStandardDeviation(), rhs.getRotationStandardDeviation());
}

//==============================================================================
//! \brief Fuzzy Compare two mounting positions \a a and \a b. NaN is equal NaN here.
//! \tparam T       The value type of the mounting position.
//! \tparam EXP     The exponent of the epsilon used for the fuzzy compare. 10^(-EXP).
//! \param[in] lhs  First vector to be compared.
//! \param[in] rhs  Second vector to be compared.
//! \return \c True if the difference between \a a and \a b is not smaller than 10^(-EXP) or if both are NaN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<typename T, uint8_t EXP>
inline bool fuzzyEqualT(const MountingPositionWithDeviation<T>& lhs, const MountingPositionWithDeviation<T>& rhs)
{
    return fuzzyEqualT<EXP>(lhs.getPosition(), rhs.getPosition()) //
           && fuzzyEqualT<EXP>(lhs.getRotation(), rhs.getRotation()) //
           && fuzzyEqualT<EXP>(lhs.getPositionStandardDeviation(), rhs.getPPositionStandardDeviation()) //
           && fuzzyEqualT<EXP>(lhs.getRotationStandardDeviation(), rhs.getRotationStandardDeviation());
}

//==============================================================================
//! \brief Stream operator for writing the mounting position to a stream.
//! \param[in, out] os                The stream, the mounting position shall be written to.
//! \param[in]      mountingPosition  The mounting position which shall be streamed.
//! \return The stream to which the mounting position was written to.
//------------------------------------------------------------------------------
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const MountingPositionWithDeviation<T>& mountingPosition)
{
    os << "(" << mountingPosition.m_position << "," << mountingPosition.m_rotation << ","
       << mountingPosition.m_positionDev << "," << mountingPosition.m_rotationDev << ")";
    return os;
}

//==============================================================================
//! \brief Stream operator for reading the mounting position from a stream.
//! \param[in, out] is                The stream, the mounting position shall be read from.
//! \param[out]     mountingPosition  The mounting position as read from the stream.
//! \return The stream from which the mounting position was read.
//!
//! \note If reading the data failed (check with /a istream::fail()) the content of the mountingPosition is undefined.
//------------------------------------------------------------------------------
template<typename T>
inline std::istream& operator>>(std::istream& is, MountingPositionWithDeviation<T>& mountingPosition)
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

    is >> mountingPosition.m_positionDev;

    is.get(charBuffer);
    if (charBuffer != ',')
    {
        // Wrong separator.
        is.setstate(std::ios::failbit);
        return is;
    }

    is >> mountingPosition.m_rotationDev;

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
// Serialization
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const MountingPositionWithDeviation<T>& p)
{
    static_assert(MountingPositionWithDeviation<T>::isSerializable(),
                  "writeBE is not implemented for given template type of MountingPositionWithDeviation");

    microvision::common::sdk::writeBE(os, p.m_position);
    microvision::common::sdk::writeBE(os, p.m_rotation);
    microvision::common::sdk::writeBE(os, p.m_positionDev);
    microvision::common::sdk::writeBE(os, p.m_rotationDev);
}

//==============================================================================

template<typename T>
inline void readBE(std::istream& is, MountingPositionWithDeviation<T>& p)
{
    static_assert(MountingPositionWithDeviation<T>::isSerializable(),
                  "readBE is not implemented for given template type of MountingPositionWithDeviation");

    microvision::common::sdk::readBE(is, p.m_position);
    microvision::common::sdk::readBE(is, p.m_rotation);
    microvision::common::sdk::readBE(is, p.m_positionDev);
    microvision::common::sdk::readBE(is, p.m_rotationDev);
}

//==============================================================================

template<typename T>
inline void writeLE(std::ostream& os, const MountingPositionWithDeviation<T>& p)
{
    static_assert(MountingPositionWithDeviation<T>::isSerializable(),
                  "writeLE is not implemented for given template type of MountingPositionWithDeviation");

    microvision::common::sdk::writeLE(os, p.m_position);
    microvision::common::sdk::writeLE(os, p.m_rotation);
    microvision::common::sdk::writeLE(os, p.m_positionDev);
    microvision::common::sdk::writeLE(os, p.m_rotationDev);
}

//==============================================================================

template<typename T>
inline void readLE(std::istream& is, MountingPositionWithDeviation<T>& p)
{
    static_assert(MountingPositionWithDeviation<T>::isSerializable(),
                  "readLE is not implemented for given template type of MountingPositionWithDeviation");

    microvision::common::sdk::readLE(is, p.m_position);
    microvision::common::sdk::readLE(is, p.m_rotation);
    microvision::common::sdk::readLE(is, p.m_positionDev);
    microvision::common::sdk::readLE(is, p.m_rotationDev);
}

//==============================================================================

template<typename T>
inline constexpr std::streamsize serializedSize(const MountingPositionWithDeviation<T>&)
{
    static_assert(MountingPositionWithDeviation<T>::isSerializable(),
                  "serializedSize is not implemented for given template type of MountingPositionWithDeviation");

    return 4 * serializedSize(Vector3<T>{});
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
