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
//! \date Aug 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

#include <microvision/common/sdk/datablocks/scan/Scan.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2321.hpp>
#include <microvision/common/sdk/datablocks/scan/special/SubScanIn2321.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/datablocks/scan/special/CorrectionSetIn2321.hpp>
#include <microvision/common/sdk/TransformationMatrix3d.hpp>

#include <microvision/common/sdk/misc/unit.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to import a third party lidar sensor raw data scan (data type 2321)
//!        from an IDC container into a generic scan object.
//------------------------------------------------------------------------------
template<>
class Importer<Scan, DataTypeId::DataType_Scan2321> : public RegisteredImporter<Scan, DataTypeId::DataType_Scan2321>
{
public:
    Importer()                = default;
    Importer(const Importer&) = delete;
    Importer& operator=(const Importer&) = delete;
    virtual ~Importer()                  = default;

public: // implements ImporterBase
    //=================================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //-------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                      const ConfigurationPtr& configuration = nullptr) const override;

    //=================================================
    //! \brief Read data from the given stream and fill the given data container (deserialization).
    //!
    //! \param[in, out] inputStream     Input data stream
    //! \param[out]     dataContainer   Output container defining the target type (might include conversion).
    //! \param[in]      dataHeader      Metadata prepended to each idc data block.
    //! \param[in]      configuration   (Optional) Configuration context for import. Default set as nullptr.
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //-------------------------------------------------
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;

public:
    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \param[in] c                   Data container to get the size from.
    //! \param[in] nbOfCorrectionSets  Number of correction sets to consider.
    //! \param[in] nbOfSubScans        Number of sub-scans to consider.
    //! \return Number of bytes used by the serialization of this data class.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c,
                                      const std::size_t nbOfCorrectionSets,
                                      const std::size_t nbOfSubScans) const;

private:
    //========================================
    //! Number of sub-scans to be used for calculating the rotational speed. This must be greater than four because
    //! for the VH64 in dual return mode, four consecutive sub-scans have the same rotational position. The higher
    //! the number the more jitters are equalized.
    //----------------------------------------
    static constexpr uint8_t rotationalSpeedCalculationWindow{32};

    // Pre-calculated sinus and cosinus value for each centi-degree.
    static constexpr uint32_t nbOfCentiDegrees{unit::angle::centiDegreesPerRotation
                                               + 1}; // 0.00 - 360.00 degrees including
    static const std::vector<float> sinLookupTableCentiDegrees;
    static const std::vector<float> cosLookupTableCentiDegrees;

private:
    //========================================
    //! \brief Get the look-up table with pre-calculated sinus values of an angle in centidegrees.
    //!
    //! \return The look-up table with pre-calculated sinus values of an angle in centidegrees.
    //----------------------------------------
    static std::vector<float> getSinLookupTableCentiDegrees();

    //========================================
    //! \brief Get the look-up table with pre-calculated cosinus values of an angle in centidegrees.
    //!
    //! \return The look-up table with pre-calculated cosinus values of an angle in centidegrees.
    //----------------------------------------
    static std::vector<float> getCosLookupTableCentiDegrees();

    //========================================
    //! \brief Calculate the rotation speed of the sensor from the timestamps of the scans.
    //!
    //! \param[in] subScans    Array with sub-scans containing the timestamps.
    //! \param[in] subScanIdx  Index in the array to start the calculation.
    //! \return The rotation speed of the sensor in centidegrees per nanosecond.
    //----------------------------------------
    static float calculateRotationalSpeed(const Scan2321::SubScanVector& subScans, const std::size_t subScanIdx);

    //========================================
    //! \brief Transform a single scan point from lidar sensor polar coordinates to cartesian coordinates.
    //!
    //! \param[in] rotationalPosition                      The rotational position in centidegrees of the sensor when
    //!                                                    the point was measured.
    //! \param[in] deviceSpecificRotationalPositionOffset  The device specific offset in the rotational position in
    //!                                                    centidegrees.
    //! \param[in] distanceValue                           The distance as measured by the sensor in units of
    //!                                                    \a distanceResolution.
    //! \param[in] rotationSpeed                           The rotation speed of the sensor in centidegrees per
    //!                                                    nanosecond.
    //! \param[in] laserTimingOffset                       The timing offset in nanoseconds relative to the lidar
    //!                                                    packet timestamp for the laser that measured this point.
    //! \param[in] distanceResolution                      The distance resolution in meters.
    //! \param[in] correctionSet                           The set of correction parameters for the laser that measured
    //!                                                    this point.
    //! \return The cartesian coordinates of this scan point.
    //----------------------------------------
    static Vector3<float> transformScanPoint(const uint16_t rotationalPosition,
                                             const uint16_t deviceSpecificRotationalPositionOffset,
                                             const uint16_t distanceValue,
                                             const float rotationSpeed,
                                             const uint32_t laserTimingOffset,
                                             const float distanceResolution,
                                             const CorrectionSetIn2321& correctionSet);

    //========================================
    //! \brief Transform a vector of scan points from lidar sensor polar to cartesian coordinates.
    //!
    //! \param[in] deviceId                                The device ID of the scanner.
    //! \param[in] deviceType                              The type of the scanner.
    //! \param[in] returnMode                              The return mode the scanner is currently working with.
    //! \param[in] calibrationData                         The set of correction parameters for all lasers.
    //! \param[in] thirdPartyVLidarSubScans                The sub-scans as measured by the sensor.
    //! \param[in] deviceSpecificRotationalPositionOffset  The device specific offset in the rotational position in
    //!                                                    centidegrees .
    //! \param[in] scannerTransformationMatrix             The matrix containing the mounting position of the scanner.
    //! \param[out] ecuScanPoints                          The list to store the transformed scan points in.
    //! \param[out] meanRotationSpeed                      The mean rotation speed across all sub-scans.
    //! \param[out] maxPointTimeOffsetUs                   The number of microseconds of the point that was measured
    //!                                                    lastly.
    //----------------------------------------
    static void transformScanPoints(const uint8_t deviceId,
                                    const ThirdPartyVLidarProtocol::DeviceType& deviceType,
                                    const ThirdPartyVLidarProtocol::ReturnMode& returnMode,
                                    const CalibrationDataIn2321& calibrationData,
                                    const Scan2321::SubScanVector& thirdPartyVLidarSubScans,
                                    const uint16_t deviceSpecificRotationalPositionOffset,
                                    const TransformationMatrix3d<float>& scannerTransformationMatrix,
                                    std::list<ScanPoint>& ecuScanPoints,
                                    float& meanRotationSpeed,
                                    uint32_t& maxPointTimeOffsetUs);

    //========================================
    //! \brief Adjust the timestamps in a scan so that the first scan point has the timestamp offset zero.
    //!
    //! \param[in,out] scan           The scan where the start and end time might be changed.
    //! \param[in,out] ecuScanPoints  The list of scan points where the time offset might be changed.
    //----------------------------------------
    static void adjustScanTimestamps(Scan& scan, std::list<ScanPoint>& ecuScanPoints);

    //========================================
    //! \brief Returns the compensated and scaled intensity of a scan point from the VH64
    //!
    //! The intensities of the VH64's scan points has to be compensated depending on the
    //! measured distance and the calibration parameters of each laser. This
    //! function uses the formula in the VH64 manual (page 45 -46, Rev K) to compute the
    //! compensated intensity from the raw intensity value in the scan point.
    //!
    //! \param[in] rawIntensity   The intensity stored in the scan point
    //! \param[in] minIntensity   The min intensity stored in the calibration of this laser
    //! \param[in] maxIntensity   The max intensity stored in the calibration of this laser
    //! \param[in] rawIntensity   The distance stored in the scan point
    //! \param[in] focalDistance  Calibration parameter for this laser
    //! \param[in] focalSlope     Calibration parameter for this laser
    //!
    //! \return                   the compensated intensity. Is scaled to the interval [0-1]
    //----------------------------------------
    static float getCompVH64Intensity(float rawIntensity,
                                      float minIntensity,
                                      float maxIntensity,
                                      float distance,
                                      float focalDistance,
                                      float focalSlope);

    //========================================
    //! \brief Transform an angle measured by a lidar scanner to an angle used in the SDK.
    //!
    //! \param[in] thirdPartyVLidarAngle             The angle in centidegrees as measured by the ThirdPartyVLidar sensor.
    //! \param[in] rotationalPositionOffset  The rotational position offset in centidegrees to consider.
    //! \return The angle in centidegrees as used in the SDK.
    //!
    //! \note This method subtracts the \a rotationalPositionOffset from the \a thirdPartyVLidarAngle, changes the rotation
    //!       from clockwise (as used by the lidar sensor) to anti-clockwise (as used by the SDK), and shifts the angle to the
    //!       range from 0 to 35999 centidegrees (if necessary).
    //----------------------------------------
    static uint16_t transformthirdPartyVLidarAngle(const uint16_t thirdPartyVLidarAngle,
                                                   const uint16_t rotationalPositionOffset);
}; //ScanImporter2321

//==============================================================================

using ScanImporter2321 = Importer<microvision::common::sdk::Scan, DataTypeId::DataType_Scan2321>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
