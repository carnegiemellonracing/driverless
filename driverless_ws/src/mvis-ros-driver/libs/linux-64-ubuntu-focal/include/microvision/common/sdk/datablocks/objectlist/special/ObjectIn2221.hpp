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

#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/Vector2.hpp>

#include <microvision/common/sdk/ObjectBasic.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ObjectIn2221 final
{
public:
    static constexpr uint16_t maxContourPoints{34};

public:
    using ContourPointVector = std::vector<Vector2<int16_t>>;

public:
    ObjectIn2221() = default;
    ObjectIn2221(const ObjectIn2221& other);
    ObjectIn2221& operator=(const ObjectIn2221& other);

    virtual ~ObjectIn2221() = default;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint16_t getObjectId() const { return m_id; }
    uint16_t getObjectAge() const { return m_age; }
    uint16_t getPredictionAge() const { return m_predictionAge; }

    uint16_t getRelativeTimestamp() const { return m_relativeTimestamp; }
    Vector2<int16_t> getReferencePoint() const { return m_refPoint; }
    Vector2<uint16_t> getReferencePointSigma() const { return m_refPointSigma; }

    Vector2<int16_t> getClosestPoint() const { return m_closestPoint; }

    Vector2<int16_t> getBoundingBoxCenter() const { return m_boundingBoxCenter; }
    uint16_t getBoundingBoxWidth() const { return m_boundingBoxWidth; }
    uint16_t getBoundingBoxLength() const { return m_boundingBoxLength; }

    Vector2<int16_t> getObjectBoxCenter() const { return m_objectBoxCenter; }
    uint16_t getObjectBoxSizeX() const { return m_objectBoxSizeX; }
    uint16_t getObjectBoxSizeY() const { return m_objectBoxSizeY; }
    int16_t getObjectBoxOrientation() const { return m_objectBoxOrientation; }

    Vector2<int16_t> getAbsoluteVelocity() const { return m_absVelocity; }
    uint16_t getAbsoluteVelocitySigmaX() const { return m_absVelSigmaX; }
    uint16_t getAbsoluteVelocitySigmaY() const { return m_absVelSigmaY; }

    Vector2<int16_t> getRelativeVelocity() const { return m_relVelocity; }

    luxObjectClass::LuxObjectClass getClassification() const { return m_class; }
    uint16_t getClassificationAge() const { return m_classAge; }
    uint16_t getClassificationCertainty() const { return m_classCertainty; }

    uint16_t getNumberOfContourPoints() const { return m_numContourPoints; }
    bool isNumContourPointsValid() const { return this->m_numContourPointsIsValid; }
    const ContourPointVector& getContourPoints() const { return m_contourPoints; }
    ContourPointVector& getContourPoints() { return m_contourPoints; }

public:
    void setObjectId(const uint16_t id) { this->m_id = id; }
    void setObjectAge(const uint16_t age) { this->m_age = age; }
    void setPredictionAge(const uint16_t predictionAge) { this->m_predictionAge = predictionAge; }
    void setRelativeTimestamp(const uint16_t relativeTimestamp) { this->m_relativeTimestamp = relativeTimestamp; }
    void setRefPoint(const Vector2<int16_t> refPoint) { this->m_refPoint = refPoint; }
    void setRefPointSigma(const Vector2<uint16_t> refPointSigma) { this->m_refPointSigma = refPointSigma; }
    void setClosestPoint(const Vector2<int16_t> closestPoint) { this->m_closestPoint = closestPoint; }
    void setBoundingBoxCenter(const Vector2<int16_t> boundingBoxCenter)
    {
        this->m_boundingBoxCenter = boundingBoxCenter;
    }
    void setBoundingBoxWidth(const uint16_t boundingBoxWidth) { this->m_boundingBoxWidth = boundingBoxWidth; }
    void setBoundingBoxLength(const uint16_t boundingBoxLength) { this->m_boundingBoxLength = boundingBoxLength; }
    void setObjectBoxCenter(const Vector2<int16_t> objectBoxCenter) { this->m_objectBoxCenter = objectBoxCenter; }
    void setObjectBoxLength(const uint16_t objectBoxLength) { this->m_objectBoxSizeX = objectBoxLength; }
    void setObjectBoxWidth(const uint16_t objectBoxWidth) { this->m_objectBoxSizeY = objectBoxWidth; }
    void setObjectBoxOrientation(const int16_t objectBoxOrientation)
    {
        this->m_objectBoxOrientation = objectBoxOrientation;
    }
    void setAbsVelocity(const Vector2<int16_t> absVelocity) { this->m_absVelocity = absVelocity; }
    void setAbsVelSigmaX(const uint16_t absVelSigmaX) { this->m_absVelSigmaX = absVelSigmaX; }
    void setAbsVelSigmaY(const uint16_t absVelSigmaY) { this->m_absVelSigmaY = absVelSigmaY; }
    void setRelVelocity(const Vector2<int16_t> relVelocity) { this->m_relVelocity = relVelocity; }
    void setClass(const luxObjectClass::LuxObjectClass cl) { this->m_class = cl; }
    void setClassAge(const uint16_t classAge) { this->m_classAge = classAge; }
    void setClassCertainty(const uint16_t classCertainty) { this->m_classCertainty = classCertainty; }
    void setNumContourPoints(const uint16_t numContourPoints);

    void setNumCoutourPointsValid(const bool valid) { this->m_numContourPointsIsValid = valid; }
    void setContourPoints(const ContourPointVector& newContourPts) { this->m_contourPoints = newContourPts; }

public:
    static float angle2rad(const int16_t ticks);

protected:
    static const uint16_t contourIsInvalid;

protected:
    uint16_t m_id{0}; //!< Id of this object from tracking.
    uint16_t m_age{0}; //!< Number of scans this object has been tracked for.

    //! Number of scans this object has currently been predicted for without measurement update. Set to 0 as soon as a
    //! measurement update is available.
    uint16_t m_predictionAge{0};

    uint16_t m_relativeTimestamp{0}; //!< Timestamp of this object relative to the scan start time in ms. [ms]
    Vector2<int16_t> m_refPoint{}; //!< Depending on tracking, this is the tracked object reference point in cm. [cm]
    Vector2<uint16_t> m_refPointSigma{}; //!<  Standard deviation of the estimated reference point position in cm. [cm]
    Vector2<int16_t> m_closestPoint{}; //!< Unfiltered position of the closest object point in cm. [cm]

    //! Center of a rectangle in the reference coordinate system containing all object points.
    Vector2<int16_t> m_boundingBoxCenter{};
    uint16_t m_boundingBoxWidth{0}; //!< Size in x direction of the rectangle containing all object points  in cm. [cm]
    uint16_t m_boundingBoxLength{0}; //!< Size in y direction of the rectangle containing all object points in cm. [cm]
    Vector2<int16_t> m_objectBoxCenter{}; //!< Object box center in the reference coordinate system in cm. [cm]
    uint16_t m_objectBoxSizeX{0}; //!< Size in x direction of the rectangle containing all object points in cm. [cm]
    uint16_t m_objectBoxSizeY{0}; //!< Size in y direction of the rectangle containing all object points in cm. [cm]
    int16_t m_objectBoxOrientation{0}; //!< Box rotated by orientation in reference coordinate system. [deg/100]
    Vector2<int16_t> m_absVelocity{}; //!< Absolut Velocity of this object in cm/s  with ego motion taken into account.
    uint16_t m_absVelSigmaX{0}; //!< Standard deviation of the estimated absolute velocity in cm/s.
    uint16_t m_absVelSigmaY{0}; //!< Standard deviation of the estimated absolute velocity in cm/s.
    Vector2<int16_t> m_relVelocity{}; //!< Velocity of this object in cm/s without ego motion compensation [cm/s]
    luxObjectClass::LuxObjectClass m_class{luxObjectClass::LuxObjectClass::Unclassified}; //!< The object class.
    uint16_t m_classAge{0}; //!<  Number of scans this object has been classified as current class  for.

    //! The class certainty.
    //!
    //! The higher this value is the more reliable is the assigned object class.
    uint16_t m_classCertainty{0};

    //!< The number of objects transmitted in this message.
    //!< Or 0xFFFFU for invalid number of points.
    uint16_t m_numContourPoints{0};

    bool m_numContourPointsIsValid{true}; //!< Flag if the number of contour points is valid. not serialized.
    ContourPointVector m_contourPoints{}; //!< Vector of contour points.
}; // ObjectIn2221

//==============================================================================

bool operator==(const ObjectIn2221& lhs, const ObjectIn2221& rhs);

inline bool operator!=(const ObjectIn2221& lhs, const ObjectIn2221& rhs) { return !(lhs == rhs); }

//==============================================================================

std::ostream& operator<<(std::ostream& os, const ObjectIn2221& luxObj);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
