//==============================================================================
//! \file
//!
//! \brief Base class for points in a point cloud.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 15, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/GpsPoint.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/ReferencePlane.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Base class for points in a point cloud.
//------------------------------------------------------------------------------
class PointBase
{
public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    PointBase() : m_epw(.0f), m_flags(0), m_rgb{255, 255, 255} {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~PointBase() = default;

public:
    //========================================
    //! \brief Calculate the serialized size of a base point of the given PointType \a type.
    //!
    //! \param[in] type  The PointType of this point.
    //! \return The size in bytes.
    //----------------------------------------
    static std::streamsize getSerializedSizeWithType_static(const PointType& type);

public:
    //========================================
    //! \brief Coordinate transformation of a given point from the \a refPlane (\a originOffset)
    //!       into global coordinates.
    //!
    //! \param[in]  refPlane     The origin reference plane.
    //! \param[in]  offset       The point's xyz coordinates in the origin reference plane.
    //! \param[out] globalPoint  On exit the point's global coordinates.
    //!
    //! \note \a P3d stand for partial 3D, altitude will taken from the point
    //!       given by offset in refPlane. If this is not set,
    //!       the altitude of the refPlane's reference point will be used.
    //----------------------------------------
    static void transformToGlobalCoordinatesP3d(const ReferencePlane& refPlane,
                                                const Vector3<float>& offset,
                                                GpsPoint& globalPoint);

    //========================================
    //! \brief Coordinate transformation of a given point from the \a refPlane (\a originOffset)
    //!       into global coordinates.
    //!
    //! \param[in]  refPlane     The origin reference plane.
    //! \param[in]  offset       The point's xyz coordinates in the origin reference plane.
    //! \param[out] globalPoint  On exit the point's global coordinates.
    //!
    //! \note \a F3d stand for full 3D. The altitude will be calculated.
    //----------------------------------------
    static void transformToGlobalCoordinatesF3d(const ReferencePlane& refPlane,
                                                const Vector3<float>& offset,
                                                GpsPoint& globalPoint);

public:
    //========================================
    //! \brief Coordinate transformation of a given point from global coordinates
    //!        into local cartesian coordinates with respect to the reference
    //!        plane \a origin.
    //!
    //! \param[in]  origin       The new reference plane with the new origin of the local coordinate system.
    //! \param[in]  globalPoint  The point's global coordinates.
    //! \param[out] offset       On exit the point's xyz coordinates in the reference plane \a origin.
    //!
    //! \note \a P3d stand for partial 3D, altitude will taken from the point
    //!       given by offset in refPlane. If this is not set,
    //!       the altitude of the refPlane GPS reference point will be used.
    //----------------------------------------
    static void transformToRelativeCoordinatesP3d(const ReferencePlane& origin,
                                                  const GpsPoint& globalPoint,
                                                  Vector3<float>& offset);

    //========================================
    //!\brief Coordinate transformation of a given point in with global coordinates
    //! into the \a refPlane.
    //!
    //! \param[in]  origin       The reference plane with the new origin of the local coordinate system.
    //! \param[in]  globalPoint  The point's global coordinates.
    //! \param[out] offset       On exit the point's xyz coordinates in the reference plane \a origin.
    //!
    //! \note \a F3d stand for full 3D. The altitude will be calculated.
    //----------------------------------------
    static void transformToRelativeCoordinatesF3d(const ReferencePlane& origin,
                                                  const GpsPoint& globalPoint,
                                                  Vector3<float>& offset);

public:
    //========================================
    //!\brief Coordinate transformation of a given point from global coordinates
    //!       into the \a originRefPlane.
    //!
    //! \param[in]  originRefPlane       The reference plane with the new origin.
    //! \param[in]  originRefPointsEcef  The \a originRefPlane's reference point
    //!                                  given as ECEF-point.
    //! \param[in]  invTargetRotMatrix   Inverse rotation between global coordinates and relative coordinates
    //!                                  in the target plane.
    //! \param[in]  globalPoint          The point's global coordinates.
    //! \param[out] offset               On exit the point's xyz coordinates in the reference plane.
    //!
    //! \note \a P3d stand for partial 3D, altitude will taken from the point
    //!       given by originOffset in originRefPlane. If this is not set,
    //!       the altitude of the originRefPlane GPS anchor point will be used.
    //----------------------------------------
    static void transformToRelativeCoordinatesP3d(const ReferencePlane& originRefPlane,
                                                  const EcefPoint& originRefPointsEcef,
                                                  const RotationMatrix3d<float>& invTargetRotMatrix,
                                                  const GpsPoint& globalPoint,
                                                  Vector3<float>& offset);

    //========================================
    //!\brief Coordinate transformation of a given point in global coordinates
    //!       into the \a originRefPlane.
    //!
    //! \param[in]  originRefPointsEcef  The \a originRefPlane's reference point
    //!                                  given as ECEF point.
    //! \param[in]  invTargetRotMatrix   Inverse rotation between global coordinates and relative coordinates
    //!                                  in the target plane.
    //! \param[in]  globalPoint          The point's global coordinates.
    //! \param[out] offset               On exit the point's xyz coordinates in the reference plane.
    //!
    //! \note \a F3d stand for full 3D. The altitude will be calculated.
    //----------------------------------------
    static void transformToRelativeCoordinatesF3d(const EcefPoint& originRefPointsEcef,
                                                  const RotationMatrix3d<float>& invTargetRotMatrix,
                                                  const GpsPoint& globalPoint,
                                                  Vector3<float>& offset);

public:
    //========================================
    //!\brief Coordinate transformation of a given point from the \a originRefPlane (\a originOffset)
    //!       to the \a targetRefPlane.
    //!\param[in]  originRefPlane  The origin reference plane.
    //!\param[in]  targetRefPlane  The target reference plane.
    //!\param[in]  originOffset    The point's xyz coordinates in the origin reference plane.
    //!\param[out] targetOffset    On exit the point's xyz coordinates in
    //!                            the target reference plane.
    //----------------------------------------
    static void transformToShiftedReferencePlaneP3d(const ReferencePlane& originRefPlane,
                                                    const ReferencePlane& targetRefPlane,
                                                    const Vector3<float>& originOffset,
                                                    Vector3<float>& targetOffset);

public:
    //========================================
    //!\brief Coordinate transformation of a given point in the \a originRefPlane (\a originOffset)
    //!       into the \a targetRefPlane.
    //!
    //! Use this method if you want to transform more than one point into the same target reference plane.
    //! Calculate \a targetRefPointsEcef using PositionWgs84::llaToEcef. This will avoid duplicate calculation
    //! other than using
    //! transformToShiftedReferencePlane(const ReferencePlane&, const ReferencePlane&, const Vector3<float>&, Vector3<float>&)
    //!
    //! \param[in]  originRefPlane       The origin reference plane.
    //! \param[in]  targetRefPlane       The target reference plane.
    //! \param[in]  targetRefPointsEcef  The ECEF coordinates and sine and cosine of the latitude and longitude
    //!                                  of the target plane reference point.
    //! \param[in]  invTargetRotMatrix   Inverse rotation between global coordinates and relative coordinates
    //!                                  in the target plane.
    //! \param[in]  originOffset         The point's xyz coordinates in the origin reference plane.
    //! \param[out] targetOffset         On exit the point's xyz coordinates in the target reference plane.
    //!
    //! \note \a P3d stand for partial 3D, altitude will taken from the point
    //!       given by originOffset in originRefPlane. If this is not set,
    //!       the altitude of the originRefPlane GPS anchor point will be used.
    //----------------------------------------
    static void transformToShiftedReferencePlaneP3d(const ReferencePlane& originRefPlane,
                                                    const ReferencePlane& targetRefPlane,
                                                    const EcefPoint& targetRefPointsEcef,
                                                    const RotationMatrix3d<float>& invTargetRotMatrix,
                                                    const Vector3<float>& originOffset,
                                                    Vector3<float>& targetOffset);

    //========================================
    //!\brief Coordinate transformation of a given point from \a originRefPlane coordinates
    //!       (\a originOffset) to \a targetRefPlane coordinates.
    //!
    //! Use this method if you want to transform more than one point into the same target reference plane.
    //! Calculate \a targetRefPointsEcef using PositionWgs84::llaToEcef. This will avoid duplicate calculation
    //! other than using
    //! transformToShiftedReferencePlane(const ReferencePlane&, const ReferencePlane&, const Vector3<float>&, Vector3<float>&)
    //!
    //! \param[in]  originRefPlane       The origin reference plane.
    //! \param[in]  targetRefPointsEcef  The ECEF coordinates and sine and cosine of the latitude and longitude
    //!                                  of the target plane reference point.
    //! \param[in]  invTargetRotMatrix   Inverse rotation between global coordinates and relative coordinates
    //!                                  in the target plane.
    //! \param[in]  originOffset         The point's xyz coordinates in the origin reference plane.
    //! \param[out] targetOffset         On exit the point's xyz coordinates in the target reference plane.
    //!
    //! \note \a F3d stand for full 3D. The altitude will be calculated.
    //----------------------------------------
    static void transformToShiftedReferencePlaneF3d(const ReferencePlane& originRefPlane,
                                                    const EcefPoint& targetRefPointsEcef,
                                                    const RotationMatrix3d<float>& invTargetRotMatrix,
                                                    const Vector3<float>& originOffset,
                                                    Vector3<float>& targetOffset);

public:
    //========================================
    //! \brief Calculate the serialized size of a base point of the given PointType \a type.
    //!
    //! \param[in] type  The PointType for which the serialized size shall be returned.
    //! \return The serialized size in bytes.
    //----------------------------------------
    virtual std::streamsize getSerializedSizeWithType(const PointType& type) const;

    //========================================
    //! \brief Deserialize the stream to point data.
    //!
    //! \param[in, out] is    Input data stream
    //! \param[in]      type  The PointType of the point to be deserialized.
    //! \return \c True if the deserialization was successful, \c false otherwise.
    //----------------------------------------
    virtual bool deserializeWithType(std::istream& is, const PointType& type);

    //========================================
    //! \brief Serialize the point data to stream
    //!
    //! \param[in, out] os    The IDC output Stream.
    //! \param[in]      type  The PointType of the point to be serialized.
    //! \return \c if the serialization was successful, \c false otherwise.
    //----------------------------------------
    virtual bool serializeWithType(std::ostream& os, const PointType& type) const;

public: // getter + setter
    //========================================
    //! \brief Get the echo pulse width of this point.
    //!
    //! \return  The echo pulse width.
    //----------------------------------------
    virtual float getEpw() const { return m_epw; }

    //========================================
    //! \brief Set the echo pulse width of this point.
    //!
    //! \param[in] epw  The new echo pulse width of this point.
    //----------------------------------------
    virtual void setEpw(const float epw) { m_epw = epw; }

    //========================================
    //! \brief Get the flags of this point.
    //!
    //! \return  The point flags.
    //----------------------------------------
    virtual uint32_t getFlags() const { return m_flags; }

    //========================================
    //! \brief Set the flags of this point.
    //!
    //! \param[in] flags  The new point flags of this point.
    //----------------------------------------
    virtual void setFlags(const uint32_t flags) { m_flags = flags; }

    //========================================
    //! \brief Get the color of this point.
    //!
    //! \return  The color.
    //----------------------------------------
    virtual std::array<uint8_t, 3> getColor() const { return m_rgb; }

    //========================================
    //! \brief Set the color of this point.
    //!
    //! \param[in] rgb  The new color of this point.
    //----------------------------------------
    virtual void setColor(const std::array<uint8_t, 3>& rgb) { m_rgb = rgb; }

private:
    float m_epw; //!< The echo pulse width of the point.
    uint32_t m_flags; //!< The point flags see PointFlag in PointCloudBase.hpp.
    std::array<uint8_t, 3> m_rgb; //!< The color values of the point.
}; // PointBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
