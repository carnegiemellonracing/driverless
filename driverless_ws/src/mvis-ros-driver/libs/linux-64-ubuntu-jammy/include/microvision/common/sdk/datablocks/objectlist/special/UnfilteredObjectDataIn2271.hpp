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
//! \date Apr 23, 2014
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/ObjectBasic.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ContourPointIn2271.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class UnfilteredObjectDataIn2271 final
{
public:
    UnfilteredObjectDataIn2271()          = default;
    virtual ~UnfilteredObjectDataIn2271() = default;

public:
    bool isValid() const { return m_isValid; }
    bool hasContourPoints() const { return m_hasContourPoints; }

    uint8_t getPriority() const { return m_priority; }
    uint16_t getRelativeTimeOfMeasure() const { return m_relativeTimeOfMeasure; }
    Vector2<int16_t> getPositionClosestObjectPoint() const { return m_positionClosestObjectPoint; }
    uint16_t getReserved() const { return m_reserved; }
    Vector2<uint16_t> getObjectBoxSize() const { return m_objectBoxSize; }
    Vector2<uint16_t> getObjectBoxSizeSigma() const { return m_objectBoxSizeSigma; }
    int16_t getObjectBoxOrientation() const { return m_objectBoxOrientation; }
    uint16_t getObjectBoxOrientationSigma() const { return m_objectBoxOrientationSigma; }
    uint8_t getObjectBoxHeight() const { return m_objectBoxHeight; }
    RefPointBoxLocation getReferencePointLocation() const { return m_referencePointLocation; }
    Vector2<int16_t> getReferencePointCoord() const { return m_referencePointCoord; }
    Vector2<uint16_t> getReferencePointCoordSigma() const { return m_referencePointCoordSigma; }
    int16_t getReferencePointPositionCorrCoeff() const { return m_referencePointPositionCorrCoeff; }
    uint8_t getExistenceProbaility() const { return m_existenceProbaility; }
    uint8_t getPossibleNbOfContourPoints() const { return m_possibleNbOfContourPoints; }

    const std::vector<ContourPointIn2271>& getContourPoints() const { return m_contourPoints; }
    std::vector<ContourPointIn2271>& getContourPoints() { return m_contourPoints; }

public:
    void setIsValid(const bool newIsValid) { m_isValid = newIsValid; }
    void setHasContourPoints(const bool newHasContourPoints) { m_hasContourPoints = newHasContourPoints; }
    void setPriority(const uint8_t newPriority) { m_priority = newPriority; }
    void setRelativeTimeOfMeasure(const uint16_t newRelativeTimeOfMeasure)
    {
        m_relativeTimeOfMeasure = newRelativeTimeOfMeasure;
    }
    void setPositionClosestObjectPoint(const Vector2<int16_t> newPositionClosestObjectPoint)
    {
        m_positionClosestObjectPoint = newPositionClosestObjectPoint;
    }
    void setReserved(const uint16_t newReserved) { m_reserved = newReserved; }
    void setObjectBoxSize(const Vector2<uint16_t> newObjectBoxSize) { m_objectBoxSize = newObjectBoxSize; }
    void setObjectBoxSizeSigma(const Vector2<uint16_t> newObjectBoxSizeSigma)
    {
        m_objectBoxSizeSigma = newObjectBoxSizeSigma;
    }
    void setObjectBoxOrientation(const int16_t newObjectBoxOrientation)
    {
        m_objectBoxOrientation = newObjectBoxOrientation;
    }
    void setObjectBoxOrientationSigma(const uint16_t newObjectBoxOrientationSigma)
    {
        m_objectBoxOrientationSigma = newObjectBoxOrientationSigma;
    }
    void setObjectBoxHeight(const uint8_t newObjectBoxHeight) { m_objectBoxHeight = newObjectBoxHeight; }
    void setReferencePointLocation(const RefPointBoxLocation newReferencePointLocation)
    {
        m_referencePointLocation = newReferencePointLocation;
    }
    void setReferencePointCoord(const Vector2<int16_t> newReferencePointCoord)
    {
        m_referencePointCoord = newReferencePointCoord;
    }
    void setReferencePointCoordSigma(const Vector2<uint16_t> newReferencePointCoordSigma)
    {
        m_referencePointCoordSigma = newReferencePointCoordSigma;
    }
    void setReferencePointPositionCorrCoeff(const int16_t newReferencePointPositionCorrCoeff)
    {
        m_referencePointPositionCorrCoeff = newReferencePointPositionCorrCoeff;
    }
    void setExistenceProbaility(const uint8_t newExistenceProbaility)
    {
        m_existenceProbaility = newExistenceProbaility;
    }
    // no setter for m_positionClosestObjectPoint, will be adjusted when serializing.

    void setPossibleNbOfContourPoints(const uint8_t possibleNbOfCtPts)
    {
        m_possibleNbOfContourPoints = possibleNbOfCtPts;
    }
    void setContourPoints(const std::vector<ContourPointIn2271>& contourPoints) { m_contourPoints = contourPoints; }

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

protected:
    bool m_isValid{false}; // not serialized
    bool m_hasContourPoints{false}; // not serialized
    uint8_t m_priority{0};
    uint16_t m_relativeTimeOfMeasure{0};
    Vector2<int16_t> m_positionClosestObjectPoint{0, 0};
    uint16_t m_reserved{0};

    Vector2<uint16_t> m_objectBoxSize{0, 0};
    Vector2<uint16_t> m_objectBoxSizeSigma{0, 0};
    int16_t m_objectBoxOrientation{0};
    uint16_t m_objectBoxOrientationSigma{0};
    uint8_t m_objectBoxHeight{0};

    RefPointBoxLocation m_referencePointLocation{RefPointBoxLocation::Unknown}; // as uint8_t
    Vector2<int16_t> m_referencePointCoord{0, 0};
    Vector2<uint16_t> m_referencePointCoordSigma{0, 0};
    int16_t m_referencePointPositionCorrCoeff{0};

    uint8_t m_existenceProbaility{0};

    mutable uint8_t m_possibleNbOfContourPoints{0};
    std::vector<ContourPointIn2271> m_contourPoints;
}; // UnfilteredObjectDataIn2271

//==============================================================================
//==============================================================================
//==============================================================================

bool operator==(const UnfilteredObjectDataIn2271& lhs, const UnfilteredObjectDataIn2271& rhs);
bool operator!=(const UnfilteredObjectDataIn2271& lhs, const UnfilteredObjectDataIn2271& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
