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
//! \date Mar 14, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/ObjectBasic.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/MeasurementSeriesIn2281.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Object with extended tracking information an dynamic Type (generic)
//------------------------------------------------------------------------------
class ObjectIn2281 final
{
public:
    enum class Flags : uint16_t
    {
        TrackedByStationaryModel = 0x0040U, //!< is object tracked using stationary model
        Mobile    = 0x0080U, //!< Has been detected/validated as mobile. (this means it has been moving at some time)
        Validated = 0x0100U //!< Object has been validated, i.e. valid enough to send out to the interface
    }; // Flags

public:
    static Vector2<float> getObjectBoxPosition(RefPointBoxLocation curRefPtLoc,
                                               const Vector2<float> refPt,
                                               const float courseAngle,
                                               const Vector2<float> objBoxSize,
                                               RefPointBoxLocation targetRefPtLoc);

public:
    ObjectIn2281();
    virtual ~ObjectIn2281() = default;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    void addMeasurement(const MeasurementIn2281& meas) { m_measurements.addMeasurement(meas); }
    void addContourPoint(const Vector2<float>& contourPoint) { m_contourPoints.push_back(contourPoint); }

public: // getter
    uint32_t getObjectId() const { return m_objectId; }
    uint16_t getFlags() const { return m_flags; }
    bool trackedByStationaryModel() const
    {
        return (m_flags & static_cast<uint16_t>(Flags::TrackedByStationaryModel))
               == static_cast<uint16_t>(Flags::TrackedByStationaryModel);
    }
    bool mobile() const
    {
        return (m_flags & static_cast<uint16_t>(Flags::Mobile)) == static_cast<uint16_t>(Flags::Mobile);
    }
    bool motionModelValidated() const
    {
        return (m_flags & static_cast<uint16_t>(Flags::Validated)) == static_cast<uint16_t>(Flags::Validated);
    }
    uint32_t getObjectAge() const { return m_objectAge; }
    NtpTime getTimestamp() const { return m_timestamp; }
    uint16_t getHiddenStatusAge() const { return m_hiddenStatusAge; }
    uint16_t getObjectPredictionAge() const { return getHiddenStatusAge(); }

    ObjectClass getClassification() const { return m_classification; }
    uint8_t getClassificationQuality() const { return m_classificationQuality; }
    uint32_t getClassificationAge() const { return m_classificationAge; }
    Vector2<float> getObjectBoxSize() const { return m_objectBoxSize; }
    Vector2<float> getObjectBoxSizeSigma() const { return m_objBoxSizeSigma; }
    float getCourseAngle() const { return m_courseAngle; }
    float getCourseAngleSigma() const { return m_courseAngleSigma; }
    Vector2<float> getRelativeVelocity() const { return m_relativeVelocity; }
    Vector2<float> getRelativeVelocitySigma() const { return m_relativeVelocitySigma; }
    Vector2<float> getAbsoluteVelocity() const { return m_absoluteVelocity; }
    Vector2<float> getAbsoluteVelocitySigma() const { return m_absoluteVelocitySigma; }
    float getObjectHeight() const { return m_objectHeight; }
    float getObjectHeightSigma() const { return m_objectHeightSigma; }
    Vector2<float> getMotionReferencePoint() const { return m_motionReferencePoint; }
    Vector2<float> getMotionReferencePointSigma() const { return m_motionReferencePointSigma; }
    float getLongitudinalAcceleration() const { return m_longitudianlAcceleration; }
    float getLongitudinalAccelerationSigma() const { return m_longitudianlAccelerationSigma; }
    float getYawRate() const { return m_yawRate; }
    float getYawRateSigma() const { return m_yawRateSigma; }
    uint8_t getNumContourPoints() const { return uint8_t(m_contourPoints.size()); }
    uint8_t getClosestContourPointIndex() const { return m_closestContourPointIndex; }
    RefPointBoxLocation getReferencePointLocation() const { return m_referencePointLocation; }
    Vector2<float> getReferencePointCoord() const { return m_referencePointCoord; }
    Vector2<float> getReferencePointCoordSigma() const { return m_referencePointCoordSigma; }
    float getReferencePointPositionCorrCoeff() const { return m_referencePointPositionCorrCoeff; }
    float getAccelerationCorrCoeff() const
    {
        return (m_measurements.contains(mkey_AccelerationCorrCoeff))
                   ? m_measurements.getMeasurement(mkey_AccelerationCorrCoeff)->getAs<float>()
                   : 0.0F;
    }
    float getVelocityCorrCoeff() const
    {
        return (m_measurements.contains(mkey_VelocityCorrCoeff))
                   ? m_measurements.getMeasurement(mkey_VelocityCorrCoeff)->getAs<float>()
                   : 0.0F;
    }

    Vector2<float> getObjectBoxPosition(const RefPointBoxLocation targetRefPtLoc) const;
    Vector2<float> getObjectBoxCenter() const;

    Vector2<float> getCenterOfGravity() const { return m_centerOfGravity; }
    uint16_t getObjectPriority() const { return m_objectPriority; }
    float getObjectExistenceMeas() const { return m_objectExistenceMeas; }

    int8_t getObjectBoxHeightOffset() const { return m_objectBoxHeightOffset; }
    float getObjectBoxHeightOffsetCm() const { return m_objectBoxHeightOffset * 4.F; }
    uint8_t getObjectBoxHeightOffsetSigma() const { return m_objectBoxHeightOffsetSigma; }

    uint8_t getReserved3() const { return m_reserved3; }
    uint8_t getReserved4() const { return m_reserved4; }

    const std::vector<Vector2<float>>& getContourPoints() const { return m_contourPoints; }

    const MeasurementSeriesIn2281& getMeasurements() const { return m_measurements; }
    const MeasurementSeriesIn2281& dynamicObjectProperties() const { return getMeasurements(); }

public: // setter
    void setObjectId(const uint32_t id) { m_objectId = id; }
    void setFlags(const uint16_t flags) { m_flags = flags; }
    void setObjectAge(const uint32_t objectAge) { m_objectAge = objectAge; }
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }
    void setHiddenStatusAge(const uint16_t hiddenStatusAge) { m_hiddenStatusAge = hiddenStatusAge; }

    void setClassification(const ObjectClass classification) { m_classification = classification; }
    void setClassificationQuality(const uint8_t classificationQuality)
    {
        m_classificationQuality = classificationQuality;
    }
    void setClassificationAge(const uint32_t classificationAge) { m_classificationAge = classificationAge; }

    void setObjectBoxSize(const Vector2<float>& objectBoxSize) { m_objectBoxSize = objectBoxSize; }
    void setObjectBoxSizeSigma(const Vector2<float>& objectBoxSizeSigma) { m_objBoxSizeSigma = objectBoxSizeSigma; }
    void setCourseAngle(const float courseAngle) { m_courseAngle = courseAngle; }
    void setCourseAngleSigma(const float courseAngleSigma) { m_courseAngleSigma = courseAngleSigma; }

    void setRelativeVelocity(const Vector2<float>& relativeVelocity) { m_relativeVelocity = relativeVelocity; }
    void setRelativeVelocitySigma(const Vector2<float>& relativeVelocitySigma)
    {
        m_relativeVelocitySigma = relativeVelocitySigma;
    }
    void setAbsoluteVelocity(const Vector2<float>& absoluteVelocity) { m_absoluteVelocity = absoluteVelocity; }
    void setAbsoluteVelocitySigma(const Vector2<float>& absoluteVelocitySigma)
    {
        m_absoluteVelocitySigma = absoluteVelocitySigma;
    }

    void setObjectHeight(const float objectHeight) { m_objectHeight = objectHeight; }
    void setObjectHeightSigma(const float objectHeightSigma) { m_objectHeightSigma = objectHeightSigma; }

    void setMotionReferencePoint(const Vector2<float>& motionReferencePoint)
    {
        m_motionReferencePoint = motionReferencePoint;
    }
    void setMotionReferencePointSigma(const Vector2<float>& motionReferencePointSigma)
    {
        m_motionReferencePointSigma = motionReferencePointSigma;
    }

    void setLongitudinalAcceleration(const float longitudinalAcceleration)
    {
        m_longitudianlAcceleration = longitudinalAcceleration;
    }
    void setLongitudinalAccelerationSigma(const float longitudinalAccelerationSigma)
    {
        m_longitudianlAccelerationSigma = longitudinalAccelerationSigma;
    }

    void setYawRate(const float yawRate) { m_yawRate = yawRate; }
    void setYawRateSigma(const float yawRateSigma) { m_yawRateSigma = yawRateSigma; }

    void setClosestContourPointIndex(const uint8_t closestContourPointIndex)
    {
        assert(closestContourPointIndex < m_contourPoints.size());
        m_closestContourPointIndex = closestContourPointIndex;
    }

    void setReferencePointLocation(const RefPointBoxLocation referencePointLocation)
    {
        m_referencePointLocation = referencePointLocation;
    }
    void setReferencePointCoord(const Vector2<float>& referencePointCoord)
    {
        m_referencePointCoord = referencePointCoord;
    }
    void setReferencePointCoordSigma(const Vector2<float>& referencePointCoordSigma)
    {
        m_referencePointCoordSigma = referencePointCoordSigma;
    }

    void setReferencePointPositionCorrCoeff(const float referencePointPositionCorrCoeff)
    {
        m_referencePointPositionCorrCoeff = referencePointPositionCorrCoeff;
    }
    void setAccelerationCorrCoeff(const float referenceAccelerationCorrCoeff);
    void setVelocityCorrCoeff(const float referenceVelocityCorrCoeff);

    void setCenterOfGravity(const Vector2<float>& centerOfGravity) { m_centerOfGravity = centerOfGravity; }
    void setObjectPriority(const uint16_t objectPriority) { m_objectPriority = objectPriority; }
    void setObjectExistenceMeas(const float objectExistenceMeas) { m_objectExistenceMeas = objectExistenceMeas; }

    void setContourPoints(const std::vector<Vector2<float>>& contourPoints) { m_contourPoints = contourPoints; }
    void setMeasurements(const MeasurementSeriesIn2281& measList) { m_measurements = measList; }

    void setObjectBoxHeightOffset(const int8_t& objectBoxHeightOffset)
    {
        m_objectBoxHeightOffset = objectBoxHeightOffset;
    }
    void setObjectBoxHeightOffsetSigma(const uint8_t& objectBoxHeightOffsetSigma)
    {
        m_objectBoxHeightOffsetSigma = objectBoxHeightOffsetSigma;
    }

    void setReserved3(const uint8_t& reserved3) { m_reserved3 = reserved3; }
    void setReserved4(const uint8_t& reserved4) { m_reserved4 = reserved4; }

public:
    Vector2<float> convertRefPoint(const RefPointBoxLocation toPos) const
    {
        return microvision::common::sdk::convertRefPoint(this->getReferencePointCoord(),
                                                         this->getCourseAngle(),
                                                         this->getObjectBoxSize(),
                                                         this->getReferencePointLocation(),
                                                         toPos);
    }

public:
    static const MeasurementKeyIn2281 mkey_oGpsImuTargetNumber; //!< Additional oGpsImu target information.
    static const MeasurementKeyIn2281 mkey_VisibilityLaser; //!< Additional oGpsImu target information.
    static const MeasurementKeyIn2281 mkey_OcclusionLaser; //!< Additional oGpsImu target information.
    static const MeasurementKeyIn2281 mkey_VisibilityDut; //!< Additional oGpsImu target information.
    static const MeasurementKeyIn2281 mkey_OcclusionDut; //!< Additional oGpsImu target information.
    static const MeasurementKeyIn2281 mkey_oGpsImuTargetType; //!< Additional oGpsImu target information.

    //========================================
    //! Acceleration correlation coefficient.
    //!
    //! Stored as float. Unitless, always between -1 and +1.
    //----------------------------------------
    static const MeasurementKeyIn2281 mkey_AccelerationCorrCoeff;

    //========================================
    //! Pearson product-moment correlation coefficient.
    //!
    //! Stored as float. Unitless, always between -1 and +1.
    //----------------------------------------
    static const MeasurementKeyIn2281 mkey_VelocityCorrCoeff;

public:
    bool operator==(const ObjectIn2281& other) const;
    bool operator!=(const ObjectIn2281& other) const { return !((*this) == other); }

protected:
    uint32_t m_objectId; //!< Id of this object from tracking.
    uint16_t m_flags; //!< The object flags. \see ObjectIn2280::Flags
    uint32_t m_objectAge; //!< Number of scans this object has been tracked for.
    NtpTime m_timestamp; //!< Timestamp of the last measurement (COG of Segment) that was used for updating this object.
    uint16_t m_hiddenStatusAge; //!< Number of update cycles that this object has only been predicted without updates.

    ObjectClass m_classification; //!< The object classification.

    //! The object classification quality.
    //!
    //! The higher this value is the more reliable is the assigned object class.
    //! Range: 0..100
    uint8_t m_classificationQuality;
    uint32_t m_classificationAge; //!< Time that this object has been classified as current class in ms. [ms]

    Vector2<float> m_objectBoxSize; //!< Size of the object box in the object coordinate system.
    Vector2<float> m_objBoxSizeSigma; //!< Standard deviation of the objectBox estimate.
    float m_courseAngle; //!< Orientation or heading of the object box [rad].
    float m_courseAngleSigma; //!< Standard deviation of the course angle.

    //! Velocity of this object in [m/s] relative to the ego vehicle in the ego vehicle coordinate system.
    Vector2<float> m_relativeVelocity;
    Vector2<float> m_relativeVelocitySigma; //!< Standard deviation of the relative velocity.
    Vector2<float> m_absoluteVelocity; //!< Absolute velocity of this object in [m/s].
    Vector2<float> m_absoluteVelocitySigma; //!< Standard deviation of the absolute velocity.

    float m_objectHeight; //!< The height of this object in [m].
    float m_objectHeightSigma; //!< The height of this object in [m].

    //! Motion reference point of this object.
    //!
    //! All motion information is related to this point.
    Vector2<float> m_motionReferencePoint;
    Vector2<float> m_motionReferencePointSigma; //!< The standard deviation of the motion reference point.

    //! Longitudinal acceleration of this object  [m/s\^2] in direction of the velocity vector.
    float m_longitudianlAcceleration;
    float m_longitudianlAccelerationSigma; //!< Standard deviation of the acceleration estimate.

    float m_yawRate; //!< Yaw rate of this object in [rad/sec]
    float m_yawRateSigma; //!< Standard deviation of the yaw rate estimate.

    // m_numContourPoints (uint8_t)
    uint8_t m_closestContourPointIndex; //!< Closest contour point of this object as index of the  point list.

    RefPointBoxLocation m_referencePointLocation; //!< Reference point location.
    Vector2<float> m_referencePointCoord; //!< The position of reference point [m]
    Vector2<float> m_referencePointCoordSigma; //!< Standard deviation of the estimated reference point position [m].
    float m_referencePointPositionCorrCoeff; //!< Pearson's product-moment coefficient.  Range: -1..1

    Vector2<float> m_centerOfGravity; //!< Center of gravity of the tracked Object.
    uint16_t m_objectPriority; //!< Determining priority. It depends on performed algorithm for tracking processing.
    float m_objectExistenceMeas; //!< Object existence measurement.

    //! Height offset: Height of lower bound of the object box from car bottom(z=0 in vehicle frame).
    //!
    //! Units are 4cm,
    //! maximum value = 5.08m,
    //! minimum value = -5.12m.
    int8_t m_objectBoxHeightOffset;
    uint8_t m_objectBoxHeightOffsetSigma; //!< Uncertainty of height offset in [cm].  max value: 2.55m
    uint8_t m_reserved3; //!< Reserved
    uint8_t m_reserved4; //!< Reserved

    std::vector<Vector2<float>> m_contourPoints; //!< Vector of contour points.
    MeasurementSeriesIn2281 m_measurements; //!<  Dynamic array of additional object properties.
}; // ObjectIn2281

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
