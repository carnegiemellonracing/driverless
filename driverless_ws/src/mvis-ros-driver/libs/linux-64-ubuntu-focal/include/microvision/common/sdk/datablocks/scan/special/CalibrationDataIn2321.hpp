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
//! \date Aug 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/CorrectionSetIn2321.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class holding all parameter needed to adjust the raw scan points received from a
//!        third party lidar scanner according to the in-factory calibration.
//------------------------------------------------------------------------------
class CalibrationDataIn2321 final
{
    friend bool operator==(const CalibrationDataIn2321& lhs, const CalibrationDataIn2321& rhs);

public:
    using CorrectionSetVector = std::vector<CorrectionSetIn2321>;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    CalibrationDataIn2321() = default;

    //========================================
    //! \brief Copy-constructor.
    //!
    //! \param[in] src  object to copy.
    //----------------------------------------
    CalibrationDataIn2321(const CalibrationDataIn2321& src) = default;

    //========================================
    //! \brief Assignment operator.
    //!
    //! \param[in] src  object to be assigned
    //! \return this
    //----------------------------------------
    CalibrationDataIn2321& operator=(const CalibrationDataIn2321& src) = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~CalibrationDataIn2321() = default;

public:
    //========================================
    //! \brief Get the size of the serialization (static version).
    //!
    //! \return Number of bytes used by the serialization of this data class.
    //----------------------------------------
    static std::streamsize getSerializedSize_static(const std::size_t nbOfCorrectionSets);

    std::streamsize getSerializedSize() const { return getSerializedSize_static(m_correctionSets.size()); }
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public: // getter
    //========================================
    //! \brief Get the resolution of the distance values in the lidar scan points.
    //!
    //! \return the distance resolution in meters.
    //----------------------------------------
    float getDistanceResolution() const { return m_distanceResolution; }

    //========================================
    //! \brief Get the number of correction sets for this scanner.
    //!
    //! \return number of correction sets (32 for VH32, 64 for VH64).
    //----------------------------------------
    uint16_t getNbOfCorrectionSets() const { return static_cast<uint16_t>(m_correctionSets.size()); }

    //========================================
    //! \brief Get the correction sets for all lasers of this scanner.
    //!
    //! \return sets of correction parameters for each laser.
    //----------------------------------------
    const CorrectionSetVector& getCorrectionSets() const { return m_correctionSets; }

    //========================================
    //! \brief Get the layer number for a given laser number.
    //!
    //! \param[in] laserIdx  laser number (0-n) to get the layer number for.
    //! \return the layer number (0-n)
    //!
    //! \note The lasers in a this lidar scanner are not numbered according to their vertical position. Thus, a mapping
    //! is created internally so that the layer number is sorted by the vertical position with zero being at
    //! the bottom.
    //----------------------------------------
    uint8_t getLayer(const uint32_t laserIdx) const;

public: // setter
    //========================================
    //! \brief Set the resolution of the distance values in the lidar sensor scan points.
    //!
    //! \param[in] distanceResolution  the new distance resolution in meters.
    //----------------------------------------
    void setDistanceResolution(const float distanceResolution) { m_distanceResolution = distanceResolution; }

    //========================================
    //! \brief Set the correction sets for all lasers of this scanner.
    //!
    //! \param[in] correctionSets  the new sets of correction parameters for each laser.
    //----------------------------------------
    void setCorrectionSets(const CorrectionSetVector& correctionSets)
    {
        m_correctionSets = correctionSets;
        createLayerMap();
    }

private:
    void createLayerMap();

private:
    float m_distanceResolution{0.002F}; //!<  Factor in [m] for distance values.
    CorrectionSetVector m_correctionSets; //!<Number of correction sets in this data set.

    // Calculated values (will not be serialized and are computed after deserialization).
    std::vector<uint8_t> m_layerMap;
}; // CalibrationDataIn2321

//==============================================================================

//==============================================================================
//! \brief Test calibration data for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise
//------------------------------------------------------------------------------
bool operator==(const CalibrationDataIn2321& lhs, const CalibrationDataIn2321& rhs);

//==============================================================================
//! \brief Test calibration data for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise
//------------------------------------------------------------------------------
inline bool operator!=(const CalibrationDataIn2321& lhs, const CalibrationDataIn2321& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
