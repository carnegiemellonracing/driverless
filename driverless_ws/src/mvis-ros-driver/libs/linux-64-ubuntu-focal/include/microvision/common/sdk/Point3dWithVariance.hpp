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
//! \date Nov 19, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/Matrix3x3.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Structure to store a 3D Point with its Variants.
// ------------------------------------------------------------------------------
template<typename T>
class Point3dWithVariance final
{
public: // type definitions
    //========================================
    //! \brief Number of values in a 3d point (incl. variance).
    //----------------------------------------
    static constexpr uint8_t nbOfValues = 12;

    //========================================
    //! \brief ValueType of the values in the vector and the matrix.
    //----------------------------------------
    using ValueType = T;

public: // constructors
    //========================================
    //! \brief Default constructor, initializes point to 0 and variance to identity matrix.
    //----------------------------------------
    Point3dWithVariance() : m_point(), m_variance() {}

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] point     Initialization vector for the point.
    //! \param[in] variance  Initialization matrix for the variance.
    //----------------------------------------
    Point3dWithVariance(const Vector3<ValueType> point, const Matrix3x3<ValueType> variance)
      : m_point{point}, m_variance{variance}
    {}

    //========================================
    //! \brief Default Destructor
    //----------------------------------------
    ~Point3dWithVariance() = default;

public: // member functions
    //========================================
    //! \brief Getter function for the point.
    //! \return The point vector.
    //! \noexcept
    //----------------------------------------
    const Vector3<ValueType>& getPoint() const noexcept { return m_point; }

    //========================================
    //! \brief Getter function for the variance.
    //! \return The variance matrix.
    //! \noexcept
    //----------------------------------------
    const Matrix3x3<ValueType>& getVariance() const noexcept { return m_variance; }

    //========================================
    //! \brief Setter function for the point
    //! \param[in] point The new vector for the point.
    //----------------------------------------
    void setPoint(const Vector3<ValueType>& point) { m_point = point; }

    //========================================
    //! \brief Setter function for the variance.
    //! \param[in] variance  The new matrix for the variance.
    //----------------------------------------
    void setVariance(const Matrix3x3<ValueType>& variance) { m_variance = variance; }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, Point3dWithVariance<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const Point3dWithVariance<TT>& value);

private: // member variables
    Vector3<ValueType> m_point;
    Matrix3x3<ValueType> m_variance;

}; // Point3dWithVariance

//==============================================================================
//! \brief Operator for comparing two points for equality.
//! \param[in] other  The point which shall be compared to the first one.
//! \return \c True, if everything is equal, \c false otherwise.
//------------------------------------------------------------------------------
template<typename T>
inline bool operator==(const Point3dWithVariance<T>& lhs, const Point3dWithVariance<T>& rhs)
{
    return (lhs.getPoint() == rhs.getPoint()) && (lhs.getVariance() == rhs.getVariance());
}

//==============================================================================
//! \brief Operator for comparing two points for inequality.
//! \param[in] other  The point which shall be compared to the first one.
//! \return \c false, if everything is equal, \c true otherwise.
//------------------------------------------------------------------------------
template<typename T>
inline bool operator!=(const Point3dWithVariance<T>& lhs, const Point3dWithVariance<T>& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
// friend functions for serialization
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const Point3dWithVariance<T>& p)
{
    microvision::common::sdk::writeBE(os, p.m_point);
    microvision::common::sdk::writeBE(os, p.m_variance);
}

//==============================================================================

template<typename T>
inline void readBE(std::istream& is, Point3dWithVariance<T>& p)
{
    microvision::common::sdk::readBE(is, p.m_point);
    microvision::common::sdk::readBE(is, p.m_variance);
}

//==============================================================================

template<typename T>
inline constexpr std::streamsize serializedSize(const Point3dWithVariance<T>&)
{
    return Point3dWithVariance<T>::nbOfValues * sizeof(T);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
