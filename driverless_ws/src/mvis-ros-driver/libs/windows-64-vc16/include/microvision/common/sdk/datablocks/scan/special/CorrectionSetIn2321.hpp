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
//! \date Jul 25, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Set of parameters used to adjust the raw scan points of the lidar scanner according to in-factory
//! calibration.
//!
//! A correction set is made of the following parameters:
//! <table>
//!   <tr><th>Parameter</th>  <th>Unit</th>  <th>Description</th>  </tr>
//!   <tr><td>rotCorrection</td>        <td>radians</td><td>The rotational correction angle for each laser, as viewed
//!                                                         from the back of the sensor. Positive factors rotate to the
//!                                                         left. Negative values rotate to the right.</td></tr>
//!   <tr><td>vertCorrection</td>       <td>radians</td><td>The vertical correction angle for each laser, as viewed
//!                                                         from the back of the sensor. Positive values have the laser
//!                                                         pointing up. Negative values have the laser pointing
//!                                                         down.</td></tr>
//!   <tr><td>distCorrection</td>       <td>meter</td>  <td>Far distance correction of each laser distance due to minor
//!                                                         laser parts' variances. Add directly to the distance value
//!                                                         read in the packet.</td></tr>
//!   <tr><td>distCorrectionX</td>      <td>meter</td>  <td>Close distance correction in X of each laser due to minor
//!                                                         laser parts' variances interpolated with far distance
//!                                                         correction then applied to measurement in X.</td></tr>
//!   <tr><td>distCorrectionY</td>      <td>meter</td>  <td>Close distance correction in Y of each laser due to minor
//!                                                         laser parts' variances interpolated with far distance
//!                                                         correction then applied to measurement in Y.</td></tr>
//!   <tr><td>vertOffsetCorrection</td> <td>meter</td>  <td>The height of each laser as measured from the bottom of the
//!                                                         base. One fixed value for all upper block lasers. Another
//!                                                         fixed value for all lower block lasers.</td></tr>
//!   <tr><td>horizOffsetCorrection</td><td>meter</td>  <td>The horizontal offset of each laser as viewed from the back
//!                                                         of the laser. Fixed positive or negative value for all
//!                                                         lasers.</td></tr>
//!   <tr><td>focalDistance</td>        <td>meter</td>  <td>The distance where the laser is most sensitive.</td></tr>
//!   <tr><td>focalSlope</td>           <td>meter</td>  <td>The factor fo the intensity compensation.</td></tr>
//! </table>
//------------------------------------------------------------------------------
class CorrectionSetIn2321 final
{
    friend bool operator==(const CorrectionSetIn2321& lhs, const CorrectionSetIn2321& rhs);

public:
    //========================================
    //! \brief Get the size of the serialization (static version).
    //!
    //! \return Number of bytes used by the serialization of this data class.
    //----------------------------------------
    static std::streamsize getSerializedSize_static();

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    CorrectionSetIn2321() = default;

    //========================================
    //! \brief Special constructor setting the vertical correction only.
    //!
    //! \param[in] vertCorrection vertical correction angle.
    //!
    //! \note This construction is used for the VH32 scanners that do not have calibration data.
    //----------------------------------------
    CorrectionSetIn2321(const float vertCorrection);

    //========================================
    //! \brief Special constructor setting the vertical, rotational and vertical offset correction.
    //!
    //! \param[in] vertCorrection        vertical correction angle.
    //! \param[in] rotCorrection         rotational correction angle.
    //! \param[in] vertOffsetCorrection  vertical offset correction
    //!
    //! \note This construction is used for lidar scanners that do not have calibration data (e.g. the VH32).
    //----------------------------------------
    CorrectionSetIn2321(const float vertCorrection, const float rotCorrection, const float vertOffsetCorrection);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~CorrectionSetIn2321() = default;

public:
    //========================================
    //! \brief Check whether this correction set is complete.
    //!
    //! \return \c True if all parameters are set to a valid value, \c false otherwise.
    //----------------------------------------
    bool isComplete() const;

public: // getter
    //========================================
    //! \brief Get the correction value for the vertical correction angle.
    //!
    //! \return the vertical correction angle in radians.
    //----------------------------------------
    float getVertCorrection() const { return m_vertCorrection; }

    //========================================
    //! \brief Get the correction value for the rotational position angle.
    //!
    //! \return the rotational position correction angle in radians.
    //----------------------------------------
    float getRotCorrection() const { return m_rotCorrection; }

    //========================================
    //! \brief Get the correction value for the distance.
    //!
    //! \return the distance correction in meters.
    //----------------------------------------
    float getDistCorrection() const { return m_distCorrection; }

    //========================================
    //! \brief Get the correction value for the vertical laser position.
    //!
    //! \return the vertical laser position correction in meters.
    //----------------------------------------
    float getVertOffsetCorrection() const { return m_vertOffsetCorrection; }

    //========================================
    //! \brief Get the correction value for the horizontal laser position.
    //!
    //! \return the horizontal laser position correction in meters.
    //----------------------------------------
    float getHorizOffsetCorrection() const { return m_horizOffsetCorrection; }

    //========================================
    //! \brief Get the pre-calculated sinus value for the rotational correction angle.
    //!
    //! \return the sinus of the rotational correction angle.
    //----------------------------------------
    float getRotCorrectionSin() const { return m_rotCorrectionSin; }

    //========================================
    //! \brief Get the pre-calculated cosinus value for the rotational correction angle.
    //!
    //! \return the cosinus of the rotational correction angle.
    //----------------------------------------
    float getRotCorrectionCos() const { return m_rotCorrectionCos; }

    //========================================
    //! \brief Get the pre-calculated sinus value for the vertical correction angle.
    //!
    //! \return the sinus of the vertical correction angle.
    //----------------------------------------
    float getVertCorrectionSin() const { return m_vertCorrectionSin; }

    //========================================
    //! \brief Get the pre-calculated cosinus value for the vertical correction angle.
    //!
    //! \return the cosinus of the vertical correction angle.
    //----------------------------------------
    float getVertCorrectionCos() const { return m_vertCorrectionCos; }

    //========================================
    //! \brief Get the pre-calculated part of the vertical offset correction used
    //!        to correct the distance in the XY plane.
    //!
    //! \return the vertical offset correction part in XY plane.
    //----------------------------------------
    float getVertOffsetCorrectionXyPlane() const { return m_vertOffsetCorrectionXyPlane; }

    //========================================
    //! \brief Get the focal distance
    //!
    //! \return the focal distance
    //----------------------------------------
    float getFocalDistance() const { return m_focalDistance; }

    //========================================
    //! \brief Get the focal slope
    //!
    //! \return the focal slope
    //----------------------------------------
    float getFocalSlope() const { return m_focalSlope; }

    //========================================
    //! \brief Get the minimum intensity
    //!
    //! \return the minimum intensity
    //----------------------------------------
    uint8_t getMinIntensity() const { return m_minIntensity; }

    //========================================
    //! \brief Get the maximum intensity
    //!
    //! \return the maximum intensity
    //----------------------------------------
    uint8_t getMaxIntensity() const { return m_maxIntensity; }

public: // setter
    //========================================
    //! \brief Set the correction value for the vertical correction angle.
    //!
    //! \param[in] vertCorrection  the new vertical correction angle in radians.
    //----------------------------------------
    void setVertCorrection(const float vertCorrection)
    {
        m_vertCorrection = vertCorrection;
        updatePreCalculatedValues();
    }

    //========================================
    //! \brief Set the correction value for the rotational position angle.
    //!
    //! \param[in] rotCorrection  the new rotational position correction angle in radians.
    //----------------------------------------
    void setRotCorrection(const float rotCorrection)
    {
        m_rotCorrection = rotCorrection;
        updatePreCalculatedValues();
    }

    //========================================
    //! \brief Set the correction value for the distance.
    //!
    //! \param[in] distCorrection  the new distance correction in meters.
    //----------------------------------------
    void setDistCorrection(const float distCorrection) { m_distCorrection = distCorrection; }

    //========================================
    //! \brief Set the correction value for the distance in X direction.
    //!
    //! \param[in] distCorrectionX  the new distance correction in meters.
    //----------------------------------------
    void setDistCorrectionX(const float distCorrectionX) { m_distCorrectionX = distCorrectionX; }

    //========================================
    //! \brief Set the correction value for the distance in Y direction.
    //!
    //! \param[in] distCorrectionY  the new distance correction in meters.
    //----------------------------------------
    void setDistCorrectionY(const float distCorrectionY) { m_distCorrectionY = distCorrectionY; }

    //========================================
    //! \brief Set the correction value for the vertical laser position.
    //!
    //! \param[in] vertOffsetCorrection  the new vertical laser position correction in meters.
    //----------------------------------------
    void setVertOffsetCorrection(const float vertOffsetCorrection)
    {
        m_vertOffsetCorrection = vertOffsetCorrection;
        updatePreCalculatedValues();
    }

    //========================================
    //! \brief Set the correction value for the horizontal laser position.
    //!
    //! \param[in] horizOffsetCorrection  the new horizontal laser position correction in meters.
    //----------------------------------------
    void setHorizOffsetCorrection(const float horizOffsetCorrection)
    {
        m_horizOffsetCorrection = horizOffsetCorrection;
    }

    //========================================
    //! \brief Set the distance where the laser is most sensitive.
    //!
    //! \param[in] focalDistance  the new focal distance in meters.
    //----------------------------------------
    void setFocalDistance(const float focalDistance) { m_focalDistance = focalDistance; }

    //========================================
    //! \brief Set the factor fo the intensity compensation.
    //!
    //! \param[in] focalSlope  the new compensation factor.
    //----------------------------------------
    void setFocalSlope(const float focalSlope) { m_focalSlope = focalSlope; }

    //========================================
    //! \brief Set the minimum intensity measured by the laser.
    //!
    //! \param[in] minIntensity  the new minimum intensity.
    //----------------------------------------
    void setMinIntensity(const uint8_t minIntensity) { m_minIntensity = minIntensity; }

    //========================================
    //! \brief Set the maximum intensity measured by the laser.
    //!
    //! \param[in] maxIntensity  the new maximum intensity.
    //----------------------------------------
    void setMaxIntensity(const uint8_t maxIntensity) { m_maxIntensity = maxIntensity; }

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

private:
    void updatePreCalculatedValues();

private:
    //! The vertical correction angle for this laser in [rad], as viewed from the back of the sensor.
    float m_vertCorrection;
    //! The rotational correction angle for this laser in [rad], as viewed from the back of the sensor.
    float m_rotCorrection;
    //! Far distance correction of this laser distance in [m] due to minor laser parts' variances.
    float m_distCorrection{0.0F};
    //! Close distance correction in X of this laser in [m] due to minor laser parts variances interpolated with far
    //! distance correction then applied to measurement in X.
    float m_distCorrectionX{0.0F};
    //! Close distance correction in Y of this laser in [m] due to minor laser parts variances interpolated with far
    //! distance correction then applied to measurement in Y.
    float m_distCorrectionY{0.0F};
    //! The height of this laser in [m] as measured from the bottom of the base.
    float m_vertOffsetCorrection;
    //! The horizontal offset of this laser in [m] as viewed from the back of the laser.
    float m_horizOffsetCorrection{0.0F};
    //! Maximum intensity distance in [m].
    float m_focalDistance{0.0F};
    //! The control intensity amount.
    float m_focalSlope{0.0F};
    //! Minimum intensity measured by this laser.
    uint8_t m_minIntensity{0};
    //! Maximum intensity measured by this laser.
    uint8_t m_maxIntensity{std::numeric_limits<uint8_t>::max()};

    // Pre-calculated values (will not be serialized and will be updated during deserialization).
    float m_rotCorrectionSin{microvision::common::sdk::NaN};
    float m_rotCorrectionCos{microvision::common::sdk::NaN};
    float m_vertCorrectionSin{microvision::common::sdk::NaN};
    float m_vertCorrectionCos{microvision::common::sdk::NaN};
    float m_vertOffsetCorrectionXyPlane{microvision::common::sdk::NaN};
}; // CorrectionSetIn2321

//==============================================================================

//==============================================================================
//! \brief Test correction sets for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise
//------------------------------------------------------------------------------
bool operator==(const CorrectionSetIn2321& lhs, const CorrectionSetIn2321& rhs);

//==============================================================================
//! \brief Test correction sets for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise
//------------------------------------------------------------------------------
inline bool operator!=(const CorrectionSetIn2321& lhs, const CorrectionSetIn2321& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
