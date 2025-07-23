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

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Information on each scan point in the format of the MOVIA scan.
//------------------------------------------------------------------------------
class ScanPointInfoListIn2340 final
{
public:
    using ScanPointInformationVector = std::vector<float>;

public:
    //========================================
    //! \brief Type of the data in the ScanPointInformationVector.
    //----------------------------------------
    enum class InformationType : uint8_t
    {
        //========================================
        //! \brief The data contains intensity, measured with the sensor in [photons].
        //----------------------------------------
        Intensity = 0x00U,

        //========================================
        //! \brief The data contains pulse width, measured with the sensor in [m].
        //----------------------------------------
        PulseWidth = 0x01U,

        //========================================
        //! \brief The data contains radial velocity (relative to sensor), measured with the sensor in [m/s].
        //----------------------------------------
        RadialVelocity = 0x02U,

        //========================================
        //! \brief The data contains the probability that a point is static in [0-1].
        //!
        //! 0: high probability to be dynamic
        //! 1: high probability to be static
        //----------------------------------------
        StaticProbability = 0x04U,

        //========================================
        //! \brief The data contains the blooming probability of each Point (0:low .. 1:high).
        //----------------------------------------
        BloomingMeasure = 0x05U,

        //========================================
        //! \brief The data contains the intershot dither measure of each point
        //!        to identify ghost reflections of strong reflectors outside the measurement range. (0:low .. 1:high)
        //----------------------------------------
        InterShotDitherMeasure = 0x06U,

        //========================================
        //! \brief The data contains the Noise floor value measured in the same unit as the intensity.
        //!        A noise floor is a property of the pixel in a way that each echo combination of
        //!        vertical ID and horizontal ID has the same noise floor.
        //----------------------------------------
        NoiseFloorValue = 0x07U,

        //========================================
        //! \brief The data contains the reference object id of each Point.
        //----------------------------------------
        MatchedToRefObjectId = 0x80U,

        //========================================
        //! \brief The data contains labels for labeling blooming from lab recordings with fixed targets (0 .. n).
        //!        The number foe each point is an or'ed combination of values where each value represents a label.
        //----------------------------------------
        BloomingLabelSet = 0x81U
    };

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScanPointInfoListIn2340() = default;

    //========================================
    //! \brief Default copy-constructor.
    //!
    //! \param[in] other  Object to create a copy from.
    //----------------------------------------
    ScanPointInfoListIn2340(const ScanPointInfoListIn2340& other) = default;

    //========================================
    //! \brief Default move-constructor.
    //----------------------------------------
    ScanPointInfoListIn2340(ScanPointInfoListIn2340&&) = default;

    //========================================
    //! \brief Default assignment operator.
    //!
    //! \param[in] other  Object to be assigned.
    //! \return Reference to \c this.
    //----------------------------------------
    ScanPointInfoListIn2340& operator=(const ScanPointInfoListIn2340& other) = default;

    //========================================
    //! \brief Default move assignment operator.
    //----------------------------------------
    ScanPointInfoListIn2340& operator=(ScanPointInfoListIn2340&& other) = default;

public: // getter
    //========================================
    //! \brief Get the type of this scan point information.
    //!
    //! \return The type of this scan point information.
    //----------------------------------------
    InformationType getInformationType() const { return m_informationType; }

    //========================================
    //! \brief Get the vector with the information for each scan point.
    //!
    //! The indices in the scan's scan point vector and the returned vector
    //! are matching.
    //!
    //! \return The vector with the information for each scan point.
    //----------------------------------------
    const ScanPointInformationVector& getScanPointInformations() const { return m_scanInfos; }

public: // setter
    //========================================
    //! \brief Set the type of this scan point information.
    //!
    //! \param[in] infoType  The new type of this scan point information.
    //----------------------------------------
    void setInformationType(const InformationType infoType) { m_informationType = infoType; }

    //========================================
    //! \brief Set the vector with the information for each scan point.
    //!
    //! The indices in the scan's scan point vector and \a scanInfos
    //! have to match.
    //!
    //! \param[in] scanInfos  The new vector with the information
    //!                       for each scan point.
    //----------------------------------------
    void setScanPointInformations(const ScanPointInformationVector& scanInfos) { m_scanInfos = scanInfos; }

    //========================================
    //! \brief Move the vector with the information for each
    //!        scan point into this scan.
    //!
    //! The indices in the scan's scan point vector and \a scanInfos
    //! have to match.
    //!
    //! \param[in] scanInfos  The new vector with the information for
    //!                       each scan point.
    //----------------------------------------
    void setScanPointInformations(ScanPointInformationVector&& scanInfos) { m_scanInfos = std::move(scanInfos); }

private:
    InformationType m_informationType{InformationType::Intensity}; //!< Type of this scan point information.
    ScanPointInformationVector m_scanInfos{}; //!< This vector contains a list of information of one specific type.

}; // ScanPointInfoListIn2340

//==============================================================================

//==============================================================================
//! \brief Write string representation for InformationType value in output stream.
//!
//! \param[in,out] outputStream     The output stream printed to.
//! \param[in]     informationType  The InformationType value.
//! \return The same output stream instance as parameter \a outputStream.
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& outputStream, const ScanPointInfoListIn2340::InformationType informationType);

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ScanPointInfoListIn2340& lhs, const ScanPointInfoListIn2340& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ScanPointInfoListIn2340& lhs, const ScanPointInfoListIn2340& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
