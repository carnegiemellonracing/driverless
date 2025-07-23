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

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ContourPointIn2271 final
{
public:
    ContourPointIn2271() = default;
    ContourPointIn2271(const Vector2<int16_t> pt);
    virtual ~ContourPointIn2271() = default;

public:
    int16_t getX() const { return m_x; }
    int16_t getY() const { return m_y; }
    uint8_t getXSigma() const { return m_xSigma; }
    uint8_t getYSigma() const { return m_ySigma; }
    int8_t getCorrCoeff() const { return m_corrCoeff; }
    uint8_t getExistenceProbability() const { return m_existenceProbability; }

public:
    void setX(const int16_t newX) { m_x = newX; }
    void setY(const int16_t newY) { m_y = newY; }
    void setXSigma(const uint8_t newXSigma) { m_xSigma = newXSigma; }
    void setYSigma(const uint8_t newYSigma) { m_ySigma = newYSigma; }
    void setCorrCoeff(const int8_t newCorrCoeff) { m_corrCoeff = newCorrCoeff; }
    void setExistenceProbability(const uint8_t newExistenceProbability)
    {
        m_existenceProbability = newExistenceProbability;
    }

public:
    static std::streamsize getSerializedSize_static() { return 8; }

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

protected:
    int16_t m_x{0}; //!<  The X coordinate of the contour point.
    int16_t m_y{0}; //!< The Y coordinate of the contour point.
    uint8_t m_xSigma{0}; //!< Standard deviation of the x value.
    uint8_t m_ySigma{0}; //!< Standard deviation of the y value.
    int8_t m_corrCoeff{0}; //!< The correction coefficient of the coordinates.
    uint8_t m_existenceProbability{0}; //!< The existence probability of the coordinates.
}; // ContourPointIn2271

//==============================================================================

bool operator==(const ContourPointIn2271& lhs, const ContourPointIn2271& rhs);
bool operator!=(const ContourPointIn2271& lhs, const ContourPointIn2271& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
