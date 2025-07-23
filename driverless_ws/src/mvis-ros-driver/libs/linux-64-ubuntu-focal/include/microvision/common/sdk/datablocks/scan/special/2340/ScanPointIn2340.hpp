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
//! \date Nov 25, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/Matrix3x3.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Scan point in the format of the MOVIA scan.
//------------------------------------------------------------------------------
class ScanPointIn2340 final
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScanPointIn2340() = default;

    //========================================
    //! \brief Default copy-constructor.
    //!
    //! \param[in] src  Object to create a copy from.
    //----------------------------------------
    ScanPointIn2340(const ScanPointIn2340& src) = default;

    //========================================
    //! \brief Default assignment operator.
    //!
    //! \param[in] src  Object to be assigned
    //! \return this
    //----------------------------------------
    ScanPointIn2340& operator=(const ScanPointIn2340& src) = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~ScanPointIn2340() = default;

public: // getter
    //========================================
    //! \brief Get the Position of the scan point in [m].
    //!
    //! \return The position of the scan point.
    //----------------------------------------
    const Vector3<float>& getPosition() const { return m_position; }

    //========================================
    //! \brief Get the time difference between scan start time of the scan and the capturing of this point in [us].
    //!
    //! \return The time difference.
    //----------------------------------------
    uint32_t getTimestampOffsetInUs() const { return m_timestampOffsetInUs; }

    //========================================
    //! \brief Get the radial distance from the virtual sensor center to the point [m].
    //!
    //! \return The radial distance.
    //----------------------------------------
    float getRadialDistance() const { return m_radialDistance; }

    //========================================
    //! \brief Get the id of the vertical angular bin.
    //!
    //! Counting starts with 0.
    //!
    //! \return The id of the vertical angular bin.
    //----------------------------------------
    uint16_t getVerticalId() const { return m_verticalId; }

    //========================================
    //! \brief Get the id of the horizontal angular bin.
    //!
    //! Counting starts with 0.
    //!
    //! \return The id of the horizontal angular bin.
    //----------------------------------------
    uint16_t getHorizontalId() const { return m_horizontalId; }

    //========================================
    //! \brief Get the id of the echo.
    //!
    //! Counting starts with 0.
    //! The echoId plus the number of following echoes is one less
    //! than the total number of echos in this beam.
    //!
    //! Echoes are sorted in ascending order by radial distance.
    //!
    //! \return The id of the echo.
    //! \sa getNumberOfFollowingEchoes
    //----------------------------------------
    uint8_t getEchoId() const { return m_echoId; }

    //========================================
    //! \brief Get the uncertainty of the position of the point.
    //!
    //! \return The uncertainty of the position.
    //----------------------------------------
    const Matrix3x3<float>& getPointUncertainty() const { return m_pointUncertainty; }

    //========================================
    //! \brief Get the probability of the existence of this point in the range [0, 1].
    //!
    //! The probability is given between 0 (low) and 1 (high).
    //!
    //! \return The existence measure of this point.
    //----------------------------------------
    float getExistenceMeasure() const { return m_existenceMeasure; }

    //========================================
    //! \brief Get the number of following echoes of the scan point.
    //!
    //! The echoId plus the number of following echoes is one less
    //! than the total number of echos in this beam.
    //!
    //! \return The number of following echoes.
    //! \sa getEchoId
    //----------------------------------------
    uint8_t getNumberOfFollowingEchoes() const { return m_nbOfFollowingEchoes; }

public: // setter
    //========================================
    //! \brief Set the Position of the scan point in [m].
    //!
    //! \param[in] position  The new position.
    //----------------------------------------
    void setPosition(const Vector3<float>& position) { m_position = position; }

    //========================================
    //! \brief Set the time difference between scan start time of the scan and the capturing of this point in [us].
    //!
    //! \param[in] timestampOffsetInUs  The new time difference.
    //----------------------------------------
    void setTimestampOffsetInUs(const uint32_t timestampOffsetInUs) { m_timestampOffsetInUs = timestampOffsetInUs; }

    //========================================
    //! \brief Set the radial distance from the virtual sensor center to the point [m].
    //!
    //! \param[in] radialDistance  The new radial distance.
    //----------------------------------------
    void setRadialDistance(const float radialDistance) { m_radialDistance = radialDistance; }

    //========================================
    //! \brief Set the id of the vertical angular bin.
    //!
    //! Counting starts with 0.
    //!
    //! \param[in] verticalId  The new id.
    //----------------------------------------
    void setVerticalId(const uint16_t verticalId) { m_verticalId = verticalId; }

    //========================================
    //! \brief Set the id of the horizontal angular bin.
    //!
    //! Counting starts with 0.
    //!
    //! \param[in] horizontalId  The new id.
    //----------------------------------------
    void setHorizontalId(uint16_t horizontalId) { m_horizontalId = horizontalId; }

    //========================================
    //! \brief Set the id of the echo.
    //!
    //! Counting starts with 0.
    //! Echoes should be sorted in ascending order by radial distance.
    //!
    //! \param[in] echoId  The new id of the echo.
    //----------------------------------------
    void setEchoId(const uint8_t echoId) { m_echoId = echoId; }

    //========================================
    //! \brief Set the uncertainty matrix of the position of this point.
    //!
    //! \param[in] pointUncertainty  The new uncertainty matrix of the position of this point.
    //----------------------------------------
    void setPointUncertainty(const Matrix3x3<float>& pointUncertainty) { m_pointUncertainty = pointUncertainty; }

    //========================================
    //! \brief Set the probability of the existence of this point in the range [0, 1].
    //!
    //! The probability is given between 0 (low) and 1 (high).
    //!
    //! \param[in] existenceMeasure  The new existence measure.
    //----------------------------------------
    void setExistenceMeasure(const float existenceMeasure) { m_existenceMeasure = existenceMeasure; }

    //========================================
    //! \brief Set the number of following echoes of the scan point.
    //!
    //! \param[in] nbOfFollowingEchoes  The new number of following echoes.
    //----------------------------------------
    void setNumberOfFollowingEchoes(const uint8_t nbOfFollowingEchoes) { m_nbOfFollowingEchoes = nbOfFollowingEchoes; }

private:
    Vector3<float> m_position{}; //!<  Position of the point in [m].

    //========================================
    //! \brief The time difference between scan start time of the scan and the capturing of this point in [us].
    //----------------------------------------
    uint32_t m_timestampOffsetInUs{0U};

    float m_radialDistance{0U}; //!< Radial distance from the virtual sensor center to the point in [m].
    uint16_t m_verticalId{0U}; //!< Id of the vertical angular bin.
    uint16_t m_horizontalId{0U}; //!< Id of the horizontal angular bin.
    uint8_t m_echoId{0U}; //!< The id of the scan point echo.
    Matrix3x3<float> m_pointUncertainty{}; //!< The uncertainty in the position of the point (will not be serialized).
    float m_existenceMeasure{0U}; //!< The probability of the existence of this point in [0-1].
    uint8_t m_nbOfFollowingEchoes{0U}; //!< The number of following echoes of the scan point.

}; // ScanPointIn2340

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ScanPointIn2340& lhs, const ScanPointIn2340& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ScanPointIn2340& lhs, const ScanPointIn2340& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
