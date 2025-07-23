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
//! \date Apr 11th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/logging/logging.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! The rays, that are sent and detected from a scala are usually distracted due to window refraction
//! or other optical issues. This class compensates those distractions using a lookup table which was
//! generated from a 3D scala simulation.
//------------------------------------------------------------------------------
class ScalaRayCorrection final
{
    enum class Apd : uint8_t
    {
        Bottom = 0,
        Middle = 1,
        Top    = 2
    };

public:
    ScalaRayCorrection();

    void process(float& hAngle, bool rearMirrorSide, uint8_t layer) const;

private:
    //!\brief Corrects horizontal and vertical angle of a single scan ray
    //!\param[in]  hAngle           Horizontal angle
    //!\param[in]  rearMirrorSide   Rear mirror side (true = -0.3 reflection)
    //!\param[in]  layer            Scan Layer (See Scan2002)
    //!\param[out] correctedVAngle  Corrected vertical angle
    //!\param[out] correctedHAngle  Corrected horizontal angle
    void correctRayAngle(float hAngle,
                         bool rearMirrorSide,
                         uint8_t layer,
                         float& correctedVAngle,
                         float& correctedHAngle) const;

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::ScalaRayCorrection";
    static microvision::common::logging::LoggerSPtr logger;

private:
    //!\brief Maps an angle to another using a polynomial which is defined by the coefficients.
    //!\param[in] inputAngle  Input angle
    //!\param[in] coeffs      Coefficients of the polynomial
    //!\return Mapped angle
    float mapAngle(float inputAngle, const std::vector<float>& coeffs) const;

    std::vector<std::vector<float>> m_polyCoefficientsVertical;
    std::vector<std::vector<float>> m_polyCoefficientsHorizontal;
    const int m_numChannels;
}; // ScalaRayCorrection

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
