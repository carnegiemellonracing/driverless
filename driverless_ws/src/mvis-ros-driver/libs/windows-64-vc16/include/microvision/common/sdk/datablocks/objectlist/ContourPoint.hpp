//==============================================================================
//! \file
//!
//! \brief Contour point (in SI units)
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ContourPoint final
{
public:
    ContourPoint() = default;
    ContourPoint(const Vector2<float> pt);
    virtual ~ContourPoint() = default;

public:
    Vector2<float> getCoordinates() const { return Vector2<float>(m_x, m_y); }
    Vector2<float> getCoordinatesSigma() const { return Vector2<float>(m_xSigma, m_ySigma); }
    float getX() const { return m_x; }
    float getY() const { return m_y; }
    float getXSigma() const { return m_xSigma; }
    float getYSigma() const { return m_ySigma; }
    float getCorrCoeff() const { return m_corrCoeff; }
    float getExistenceProbability() const { return m_existenceProbability; }

public:
    void setCoordinates(const Vector2<float>& coords)
    {
        m_x = coords.getX();
        m_y = coords.getY();
    }
    void setCoordinatesSigma(const Vector2<float>& coordsSigma)
    {
        m_xSigma = coordsSigma.getX();
        m_ySigma = coordsSigma.getY();
    }
    void setX(const float newX) { m_x = newX; }
    void setY(const float newY) { m_y = newY; }
    void setXSigma(const float newXSigma) { m_xSigma = newXSigma; }
    void setYSigma(const float newYSigma) { m_ySigma = newYSigma; }
    void setCorrCoeff(const float newCorrCoeff) { m_corrCoeff = newCorrCoeff; }
    void setExistenceProbability(const float newExistenceProbability)
    {
        m_existenceProbability = newExistenceProbability;
    }

protected:
    float m_x{NaN};
    float m_y{NaN};
    float m_xSigma{NaN};
    float m_ySigma{NaN};
    float m_corrCoeff{NaN};
    float m_existenceProbability{NaN};
}; // ContourPoint

//==============================================================================
//==============================================================================
//==============================================================================

bool operator==(const ContourPoint& lhs, const ContourPoint& rhs);
bool operator!=(const ContourPoint& lhs, const ContourPoint& rhs);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
