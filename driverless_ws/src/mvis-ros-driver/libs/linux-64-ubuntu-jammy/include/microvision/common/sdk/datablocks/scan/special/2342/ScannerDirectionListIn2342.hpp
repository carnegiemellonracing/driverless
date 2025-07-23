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
//! \date Jan 18, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/MountingPositionWithDeviation.hpp>
#include <microvision/common/sdk/Vector2.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Beam directions of each pixels of a MOVIA scan ner.
//------------------------------------------------------------------------------
class ScannerDirectionListIn2342 final
{
public:
    static constexpr uint16_t nbOfDirections{10240}; //!< Number of all (nbOfRows) * (nbOfPoints) pixels directions.

public:
    //========================================
    //! \brief Array of all 80*128 pixels directions.
    //!
    //! Direction with x = azimuth and y = elevation angle in rad in ISMC.
    //----------------------------------------
    using PixelDirectionArray = std::array<Vector2<float>, nbOfDirections>;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScannerDirectionListIn2342() = default;

    //========================================
    //! \brief Default copy-constructor.
    //!
    //! \param[in] src  Object to create a copy from.
    //----------------------------------------
    ScannerDirectionListIn2342(const ScannerDirectionListIn2342& src) = default;

    //========================================
    //! \brief Default assignment operator.
    //!
    //! \param[in] src  Object to be assigned.
    //! \return A reference to \c this.
    //----------------------------------------
    ScannerDirectionListIn2342& operator=(const ScannerDirectionListIn2342& src) = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ScannerDirectionListIn2342() = default;

public:
    //========================================
    //! \brief Get the divergence angle of the scanner in mm/m .
    //!
    //! \return The divergence angle of the scanner.
    //----------------------------------------
    float getDivergenceAngleInMillimetersPerMeter() const { return m_divergenceAngle; }

    //========================================
    //! \brief Get the divergence offset of the scanner in mm.
    //!
    //! \return The divergence offset of the scanner.
    //----------------------------------------
    float getDivergenceOffsetInMillimeter() const { return m_divergenceOffset; }

    //========================================
    //! \brief Get the radial distance standard deviation of the scanner in cm.
    //!
    //! \return The radial distance standard deviation of the scanner.
    //----------------------------------------
    MICROVISION_SDK_DEPRECATED float getRadialDistanceStandardDeviationInCentimeter() const
    {
        return m_radialDistanceStdDev;
    }

    //========================================
    //! \brief Get the beam directions for all pixels.
    //!
    //! \return The beam directions of the scanner pixels.
    //----------------------------------------
    const PixelDirectionArray& getPixelDirections() const { return m_directions; }

    //========================================
    //! \brief Get the beam directions for all pixels.
    //!
    //! \return The beam directions of the scanner pixels.
    //----------------------------------------
    PixelDirectionArray& getPixelDirections() { return m_directions; }

public: // setter
    //========================================
    //! \brief Set the divergence angle of the scanner in mm/m.
    //!
    //! \param[in] divergenceAngle  The new divergence angle.
    //----------------------------------------
    void setDivergenceAngleInMillimetersPerMeter(const float divergenceAngle) { m_divergenceAngle = divergenceAngle; }

    //========================================
    //! \brief Set the divergence offset of the scanner in mm.
    //!
    //! \param[in] divergenceOffset  The new divergence offset.
    //----------------------------------------
    void setDivergenceOffsetInMillimeter(const float divergenceOffset) { m_divergenceOffset = divergenceOffset; }

    //========================================
    //! \brief Set the standard deviation of the radial distance of the echo in cm.
    //!
    //! \param[in] radialDistanceStdDev  The new standard deviation of the radial distance.
    //!
    //! \deprecated No longer relevant.
    //----------------------------------------
    MICROVISION_SDK_DEPRECATED void setRadialDistanceStandardDeviationInCentimeter(const float radialDistanceStdDev)
    {
        m_radialDistanceStdDev = radialDistanceStdDev;
    }

    //========================================
    //! \brief Set the pixel directions of the scanner pixels.
    //!
    //! \param[in] directions  The new pixel directions.
    //----------------------------------------
    void setPixelDirections(const PixelDirectionArray& directions) { m_directions = directions; }

private:
    float m_divergenceAngle{}; //!< The divergenceAngle in mm/m.

    //========================================
    //! Beam divergence at the distance of the lens (lens level) in mm.
    //!
    //! \attention It is assumed, that the divergence is the same for azimuth and elevation.
    //----------------------------------------
    float m_divergenceOffset{};

    //========================================
    //!< The standard deviation of the radial distance.
    //! \deprecated No longer relevant.
    //----------------------------------------
    float m_radialDistanceStdDev{};

    PixelDirectionArray m_directions{}; //!< The beam directions for all pixels.

}; // ScannerDirectionListIn2342

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ScannerDirectionListIn2342& lhs, const ScannerDirectionListIn2342& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ScannerDirectionListIn2342& lhs, const ScannerDirectionListIn2342& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
