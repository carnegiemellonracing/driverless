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
//! \date Apr 26, 2012
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/ObjectBasic.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ObjectIn2225 final
{
public:
    ObjectIn2225();
    ObjectIn2225(const ObjectIn2225& src);
    virtual ~ObjectIn2225() = default;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint16_t getObjectId() const { return m_id; }
    uint16_t getReserved() const { return m_reserved; }
    uint32_t getObjectAge() const { return m_age; }

    NtpTime getTimestamp() const { return m_timestamp; }
    uint16_t getHiddenStatusAge() const { return m_hiddenStatusAge; }

    ObjectClass getClassification() const { return m_class; }
    uint8_t getClassificationCertainty() const { return m_classCertainty; }
    uint32_t getClassificationAge() const { return m_classAge; }

    Vector2<float> getBoundingBoxCenter() { return m_boundingBoxCenter; }
    const Vector2<float>& getBoundingBoxCenter() const { return m_boundingBoxCenter; }
    Vector2<float> getBoundingBoxSize() { return m_boundingBoxSize; }
    const Vector2<float>& getBoundingBoxSize() const { return m_boundingBoxSize; }

    Vector2<float> getObjectBoxCenter() { return m_objectBoxCenter; }
    const Vector2<float>& getObjectBoxCenter() const { return m_objectBoxCenter; }
    Vector2<float> getObjectBoxSigma() { return m_objectBoxSigma; }
    const Vector2<float>& getObjectBoxSigma() const { return m_objectBoxSigma; }
    Vector2<float> getObjectBoxSize() { return m_objectBoxSize; }
    const Vector2<float>& getObjectBoxSize() const { return m_objectBoxSize; }
    uint64_t getReserved2() const { return m_reserved2; }

    float getObjectBoxOrientation() const { return m_objectBoxOrientation; }
    float getObjectBoxOrientationSigma() const { return m_objectBoxOrientationSigma; }

    Vector2<float> getRelativeVelocity() { return m_relVelocity; }
    const Vector2<float>& getRelativeVelocity() const { return m_relVelocity; }
    Vector2<float> getRelativeVelocitySigma() { return m_relVelocitySigma; }
    const Vector2<float>& getRelativeVelocitySigma() const { return m_relVelocitySigma; }

    Vector2<float> getAbsoluteVelocity() { return m_absVelocity; }
    const Vector2<float>& getAbsoluteVelocity() const { return m_absVelocity; }
    Vector2<float> getAbsoluteVelocitySigma() { return m_absVelocitySigma; }
    const Vector2<float>& getAbsoluteVelocitySigma() const { return m_absVelocitySigma; }
    float getOrientation() const { return this->m_orientation; }

    uint8_t getNumberOfContourPoints() const { return m_numContourPoints; }
    std::vector<Vector2<float>>& getContourPoints() { return m_contourPoints; }
    const std::vector<Vector2<float>>& getContourPoints() const { return m_contourPoints; }

    uint8_t getIndexOfClosestPoint() const { return m_indexOfClosedPoint; }

public:
    void setObjectId(const uint16_t newObjectId) { m_id = newObjectId; }
    void setReserved(const uint16_t newReserved) { m_reserved = newReserved; }
    void setObjectAge(const uint32_t newObjectAge) { m_age = newObjectAge; }

    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }
    void setHiddenStatusAge(const uint16_t newHiddenStatusAge) { m_hiddenStatusAge = newHiddenStatusAge; }

    void setClassification(const ObjectClass newClassification) { m_class = newClassification; }
    void setClassificationCertainty(const uint8_t newClassificationCertainty)
    {
        m_classCertainty = newClassificationCertainty;
    }
    void setClassificationAge(const uint32_t newClassificationAge) { m_classAge = newClassificationAge; }

    void setBoundingBoxCenter(const Vector2<float>& newBoundingBoxCenter)
    {
        m_boundingBoxCenter = newBoundingBoxCenter;
    }
    void setBoundingBoxSize(const Vector2<float>& newBoundingBoxSize) { m_boundingBoxSize = newBoundingBoxSize; }
    void setReserved2(const uint64_t& newReserved2) { m_reserved2 = newReserved2; }

    void setObjectBoxCenter(const Vector2<float>& newObjectBoxCenter) { m_objectBoxCenter = newObjectBoxCenter; }
    void setObjectBoxSigma(const Vector2<float>& newObjectBoxSigma) { m_objectBoxSigma = newObjectBoxSigma; }
    void setObjectBoxSize(const Vector2<float>& newObjectBoxSize) { m_objectBoxSize = newObjectBoxSize; }

    void setObjBoxOrientation(const float newObjBoxOrientation) { m_objectBoxOrientation = newObjBoxOrientation; }
    void setObjBoxOrientationSigma(const float newObjBoxOrientationSigma)
    {
        m_objectBoxOrientationSigma = newObjBoxOrientationSigma;
    }

    void setRelativeVelocity(const Vector2<float>& newRelativeVelocity) { m_relVelocity = newRelativeVelocity; }
    void setRelativeVelocitySigma(const Vector2<float>& newRelativeVelocitySigma)
    {
        m_relVelocitySigma = newRelativeVelocitySigma;
    }

    void setAbsoluteVelocity(const Vector2<float>& newAbsoluteVelocity) { m_absVelocity = newAbsoluteVelocity; }
    void setAbsoluteVelocitySigma(const Vector2<float>& newAbsoluteVelocitySigma)
    {
        m_absVelocitySigma = newAbsoluteVelocitySigma;
    }
    void setOrientation(const float newOrientation) { this->m_orientation = newOrientation; }

    void setNumberOfContourPoints(const uint8_t newNumberOfContourPoints)
    {
        m_numContourPoints = newNumberOfContourPoints;
    }
    void setContourPoints(const std::vector<Vector2<float>>& newContourPoints) { m_contourPoints = newContourPoints; }

    void setIndexOfClosestPoint(const uint8_t newIndexOfClosestPoint) { m_indexOfClosedPoint = newIndexOfClosestPoint; }

public:
    bool operator==(const ObjectIn2225& other) const;
    bool operator!=(const ObjectIn2225& other) const { return !((*this) == other); }

protected:
    static const int nbOfBytesInReserved3 = 18;

protected:
    uint16_t m_id; //!< Id of this object from tracking.
    uint16_t m_reserved; //!< Reserved bytes.
    uint32_t m_age; //!< Number of scans this object has been tracked for.
    NtpTime m_timestamp; //!< Time when this object was observed.
    uint16_t m_hiddenStatusAge; //!< Number of scans this object has only been predicted without measurement updates.

    ObjectClass m_class; //!< The Object classification.

    //! The certainty of the object classification.
    //!
    //! The higher this value is the more reliable is the assigned object class.
    uint8_t m_classCertainty;
    uint32_t m_classAge; //!< Number of scans this object has been classified as current class.

    Vector2<float> m_boundingBoxCenter; //!< Center point of the bounding box of this object.
    Vector2<float> m_boundingBoxSize; //!< Size of the bounding box

    Vector2<float> m_objectBoxCenter; //!< Center point (tracked) of this object.
    Vector2<float> m_objectBoxSigma; //!< Standard deviation of the object box center point.
    Vector2<float> m_objectBoxSize; //!<  Size of the object box in the object coordinate system.

    uint64_t m_reserved2; //!< Reserved bytes.

    float m_objectBoxOrientation; //!< Orientation or heading (Yaw angle) of the object in radians. [rad]
    float m_objectBoxOrientationSigma; //!< Standard deviation of yaw angle.

    Vector2<float> m_relVelocity; //!< Velocity of this object in m/s relative to the ego vehicle. [m/s]
    Vector2<float> m_relVelocitySigma; //!< Standard deviation of the relative velocity.

    Vector2<float> m_absVelocity; //!<  Velocity of this object in m/s with ego motion taken into account. [m/s]
    Vector2<float> m_absVelocitySigma; //!< Standard deviation of the absolute velocity.

    char m_reserved3[nbOfBytesInReserved3]; //!< Reserved bytes.

    uint8_t m_numContourPoints; //!< Number of contour points transmitted for this object.
    uint8_t m_indexOfClosedPoint; //!< Closes contour point of this object as index of the point list.
    std::vector<Vector2<float>> m_contourPoints; //!< Vector of contour points in m. [m]

protected: // not serialized data
    float m_orientation;
}; // ObjectIn2225

//==============================================================================

std::ostream& operator<<(std::ostream& os, const ObjectIn2225& o);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
