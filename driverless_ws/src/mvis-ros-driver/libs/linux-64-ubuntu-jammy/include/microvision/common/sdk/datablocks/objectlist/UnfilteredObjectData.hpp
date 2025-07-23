//==============================================================================
//! \file
//!
//! \brief Unfiltered object data (in SI units)
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 9, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ContourPoint.hpp>
#include <microvision/common/sdk/ObjectBasic.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class UnfilteredObjectData final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    UnfilteredObjectData()          = default;
    virtual ~UnfilteredObjectData() = default;

public:
    uint8_t getPriority() const { return m_priority; }
    NtpTime getTimestamp() const { return m_timestamp; }
    Vector2<float> getPositionClosestObjectPoint() const { return m_positionClosestObjectPoint; }
    Vector2<float> getObjectBoxSize() const { return m_objectBoxSize; }
    Vector2<float> getObjectBoxSizeSigma() const { return m_objectBoxSizeSigma; }
    float getObjectBoxHeight() const { return m_objectBoxHeight; }
    float getCourseAngle() const { return m_courseAngle; }
    float getCourseAngleSigma() const { return m_courseAngleSigma; }
    RefPointBoxLocation getReferencePointLocation() const { return m_referencePointLocation; }
    Vector2<float> getReferencePointCoord() const { return m_referencePointCoord; }
    Vector2<float> getReferencePointCoordSigma() const { return m_referencePointCoordSigma; }
    float getReferencePointCoordCorrCoeff() const { return m_referencePointCoordCorrCoeff; }
    float getExistenceProbability() const { return m_existenceProbability; }

    uint8_t getNumContourPoints() const { return static_cast<uint8_t>(m_contourPoints.size()); }
    const std::vector<ContourPoint>& getContourPoints() const { return m_contourPoints; }
    std::vector<ContourPoint>& getContourPoints() { return m_contourPoints; }

public:
    void setPriority(const uint8_t newPriority) { m_priority = newPriority; }
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }
    void setPositionClosestObjectPoint(const Vector2<float> newPositionClosestObjectPoint)
    {
        m_positionClosestObjectPoint = newPositionClosestObjectPoint;
    }
    void setObjectBoxSize(const Vector2<float> newObjectBoxSize) { m_objectBoxSize = newObjectBoxSize; }
    void setObjectBoxSizeSigma(const Vector2<float> newObjectBoxSizeSigma)
    {
        m_objectBoxSizeSigma = newObjectBoxSizeSigma;
    }
    void setObjectBoxHeight(const float newObjectBoxHeight) { m_objectBoxHeight = newObjectBoxHeight; }
    void setCourseAngle(const float newCourseAngle) { m_courseAngle = newCourseAngle; }
    void setCourseAngleSigma(const float newCourseAngleSigma) { m_courseAngleSigma = newCourseAngleSigma; }
    void setReferencePointLocation(const RefPointBoxLocation newReferencePointLocation)
    {
        m_referencePointLocation = newReferencePointLocation;
    }
    void setReferencePointCoord(const Vector2<float> newReferencePointCoord)
    {
        m_referencePointCoord = newReferencePointCoord;
    }
    void setReferencePointCoordSigma(const Vector2<float> newReferencePointCoordSigma)
    {
        m_referencePointCoordSigma = newReferencePointCoordSigma;
    }
    void setReferencePointCoordCorrCoeff(const float newReferencePointCoordCorrCoeff)
    {
        m_referencePointCoordCorrCoeff = newReferencePointCoordCorrCoeff;
    }
    void setExistenceProbability(const float newExistenceProbaility)
    {
        m_existenceProbability = newExistenceProbaility;
    }

protected:
    uint8_t m_priority{0xFF};
    NtpTime m_timestamp{0};
    Vector2<float> m_positionClosestObjectPoint{NaN, NaN};

    Vector2<float> m_objectBoxSize{NaN, NaN};
    Vector2<float> m_objectBoxSizeSigma{NaN, NaN};
    float m_objectBoxHeight{NaN};
    float m_courseAngle{NaN};
    float m_courseAngleSigma{NaN};

    RefPointBoxLocation m_referencePointLocation{RefPointBoxLocation::Unknown};
    Vector2<float> m_referencePointCoord{NaN, NaN};
    Vector2<float> m_referencePointCoordSigma{NaN, NaN};
    float m_referencePointCoordCorrCoeff{NaN};

    float m_existenceProbability{NaN};

    std::vector<ContourPoint> m_contourPoints;
}; // UnfilteredObjectData

//==============================================================================
//==============================================================================
//==============================================================================

bool operator==(const UnfilteredObjectData& lhs, const UnfilteredObjectData& rhs);
bool operator!=(const UnfilteredObjectData& lhs, const UnfilteredObjectData& rhs);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
