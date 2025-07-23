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
//! \date Feb 10, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/ScannerInfo.hpp>
#include <microvision/common/sdk/Math.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Provides properties used for scanners.
//------------------------------------------------------------------------------
class ScannerProperties final
{
public:
    //========================================
    //! \brief Get the properties for the scanner type
    //!        of the given ScannerTypeId.
    //!
    //! \param[in] scannerTypeId  Scanner type to receive the
    //!                           properties for.
    //! \return ScannerProperties for scanner with scanner type
    //!         \a scannerTypeId. If the ScannerProperties are
    //!         not registered for the given \a scannerTypeId,
    //!         the ScannerProperties for #ScannerType_Invalid
    //!         will be returned.
    //!         I.e. you can check whether you got valid
    //!         ScannerProperties by comparing \a scannerTypeId
    //!         and getScannerProperties(scannerTypeId).getScannerType().
    //----------------------------------------
    static const ScannerProperties& getScannerProperties(const uint16_t scannerTypeId);

    //========================================
    //! \brief Register additional scanner properties.
    //!
    //! To call this method is only necessary to register
    //! additional scanner properties.
    //! Currently the following ScannerTypes are already
    //! registered: #ScannerType_Invalid, #ScannerType_Ecu,
    //! #ScannerType_Lux, #ScannerType_Lux4, #ScannerType_MiniLux,
    //! #ScannerType_ScalaB2.
    //!
    //! \param[in] sp  ScannerProperties to be registered for the
    //!                ScannerTypeId given inside \a sp.
    //! \return If any error occur a value different from 0 will be
    //!         returned.
    //! \retval 0  No error occurred, ScannerProperties are registered.
    //! \retval 1  ScannerProperties for the given ScannerTypeId has
    //!            already been registered. Registration failed.
    //----------------------------------------
    static int registerAdditionalScanner(const ScannerProperties& sp);

protected:
    //========================================
    //! \brief Defalut constructor.
    //----------------------------------------
    ScannerProperties() = default;

public:
    //========================================
    //! \brief Creates ScannerProperties.
    //!
    //! \param[in] scannerType               ScannerType of this properties.
    //! \param[in] nbOfLayers                Number of Scan Layers.
    //! \param[in] maxNbOfEchos              Maximal Echos per ScanPoint.
    //! \param[in] verticalBeamDivergence    Vertical beam divergence of the Scanner.
    //! \param[in] horizontalBeamDivergence  Horizontal beam divergence of the Scanner.
    //! \param[in] verticalResolution        Vertical resolution of the Scanner.
    //! \param[in] distanceAccuracy          Distance accuracy of the Scanner.
    //! \param[in] normTargetRange           Norm target range.
    //! \param[in] minScanAngle              The minimal Scan angle.
    //! \param[in] maxScanAngle              The maximal Scan angle.
    //----------------------------------------
    ScannerProperties(const ScannerType scannerType,
                      const uint8_t nbOfLayers,
                      const uint8_t maxNbOfEchos,
                      const float verticalBeamDivergence,
                      const float horizontalBeamDivergence,
                      const float verticalResolution,
                      const float distanceAccuracy,
                      const float normTargetRange,
                      const float minScanAngle,
                      const float maxScanAngle);

public: // getter
    //========================================
    //! \brief Get the ScannerType of this properties.
    //! \return The ScannerType.
    //----------------------------------------
    ScannerType getScannerType() const { return m_scannerType; }

    //========================================
    //! \brief Get the number of Scan Layers.
    //! \return The number of Scan Layers.
    //----------------------------------------
    uint8_t getNbOfLayers() const { return m_nbOfLayers; }

    //========================================
    //! \brief Get the maximal Echos per ScanPoint.
    //! \return The maximal Echos per ScanPoint.
    //----------------------------------------
    uint8_t getMaxNbOfEchos() const { return m_maxNbOfEchos; }

    //========================================
    //! \brief Get the vertical beam divergence of the Scanner.
    //! \return The vertical beam divergence of the Scanner.
    //----------------------------------------
    float getVerticalBeamDivergence() const { return m_verticalBeamDivergence; }

    //========================================
    //! \brief Get the horizontal beam divergence of the Scanner.
    //! \return The horizontal beam divergence of the Scanner.
    //----------------------------------------
    float getHorizontalBeamDivergence() const { return m_horizontalBeamDivergence; }

    //========================================
    //! \brief Get the vertical resolution of the Scanner.
    //! \return The vertical resolution of the Scanner.
    //----------------------------------------
    float getVerticalResolution() const { return m_verticalResolution; }

    //========================================
    //! \brief Get the distance accuracy of the Scanner.
    //! \return The distance accuracy of the Scanner.
    //----------------------------------------
    float getDistanceAccuracy() const { return m_distanceAccuracy; }

    //========================================
    //! \brief Get the norm target range.
    //! \return The norm target range.
    //----------------------------------------
    float getNormTargetRange() const { return m_normTargetRange; }

    //========================================
    //! \brief Get the minimal Scan angle.
    //! \return The minimal Scan angle.
    //----------------------------------------
    float getMinScanAngle() const { return m_minScanAngle; }

    //========================================
    //! \brief Get the maximal Scan angle.
    //! \return The maximal Scan angle.
    //----------------------------------------
    float getMaxScanAngle() const { return m_maxScanAngle; }

private:
    //========================================
    //! \brief Register standard scanner properties.
    //!
    //! Currently the following ScannerTypes will be registered:
    //!     #ScannerType_Invalid,
    //!     #ScannerType_Ecu,
    //!     #ScannerType_Lux,
    //!     #ScannerType_Lux4,
    //!     #ScannerType_MiniLux,
    //!     #ScannerType_ScalaB2,
    //!     #ThirdPartyVLidarVH32,
    //!     #ThirdPartyVLidarVH64,
    //!     #ThirdPartyVLidarVS128,
    //!     #ThirdPartyHLidar40,
    //!     #ThirdPartyHLidar40P,
    //!     #ThirdPartyHLidar64,
    //!     #ThirdPartyOLidar32
    //----------------------------------------
    static int registerStdScanners();

    //========================================
    //! \brief Template function to create default ScannerProperties
    //----------------------------------------
    template<uint16_t ScannerTypeId>
    static ScannerProperties createScannerProperties();

private:
    //========================================
    //! \brief Unordered map of registered ScannerProperties.
    //----------------------------------------
    static std::unordered_map<uint16_t, std::unique_ptr<ScannerProperties>> propertiesMap;

    //========================================
    //! \brief Registration Error of the standard registration.
    //----------------------------------------
    static int registrationError;

private:
    ScannerType m_scannerType; //< ScannerType of this properties.
    uint8_t m_nbOfLayers; //< Number of Scan Layers.
    uint8_t m_maxNbOfEchos; //< Maximal Echos per ScanPoint.
    float m_verticalBeamDivergence; //< Vertical beam divergence of the Scanner.
    float m_horizontalBeamDivergence; //< Horizontal beam divergence of the Scanner.
    float m_verticalResolution; //< Vertical resolution of the Scanner.
    float m_distanceAccuracy; //< Distance accuracy of the Scanner.
    float m_normTargetRange; //< Norm target range.
    float m_minScanAngle; //< The minimal Scan angle.
    float m_maxScanAngle; //< The maximal Scan angle.
}; // ScannerProperties

//==============================================================================
// ToDo Add ScannerProperties for:
//  - MiniLux, LuxHr
//  - ThirdpartySLidar Lms200, Lms100, Jeff, TiM300, Mrs1000, Lms1000
//  - ThirdPartyVLidarVV16
//==============================================================================

template<>
inline ScannerProperties ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::Invalid)>()
{
    return ScannerProperties(ScannerType::Invalid, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN, NaN);
}

//==============================================================================

template<>
inline ScannerProperties ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::Ecu)>()
{
    return ScannerProperties(ScannerType::Ecu, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN, NaN);
}

//==============================================================================

template<>
inline ScannerProperties ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::Lux)>()
{
    return ScannerProperties(
        ScannerType::Lux, 4, 3, 0.8F * deg2radf, 0.08F * deg2radf, 0.8F * deg2radf, 0.1F, 50.0F, NaN, NaN);
}

//==============================================================================

template<>
inline ScannerProperties ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::Lux4)>()
{
    return ScannerProperties(
        ScannerType::Lux4, 4, 3, 0.8F * deg2radf, 0.08F * deg2radf, 0.8F * deg2radf, 0.1F, 50.0F, NaN, NaN);
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartySLidarL500)>()
{
    // The ThirdPartySLidarL500 uses only one layer and no vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartySLidarL500, // scannerType
                             1, // nbOfLayers
                             5, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             NaN, // distanceAccuracy
                             NaN, // normTargetRange
                             -95.0F * deg2radf, // minScanAngle
                             +95.0F * deg2radf); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ScalaB2)>()
{
    return ScannerProperties(ScannerType::ScalaB2,
                             3,
                             3,
                             0.8F * deg2radf,
                             0.08F * deg2radf,
                             0.8F * deg2radf,
                             0.1F,
                             50.0F,
                             -72.5F * deg2radf,
                             72.5F * deg2radf);
}

//==============================================================================

template<>
inline ScannerProperties ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::Movia)>()
{
    // The MOVIA properties are not constant. (Mock Setup for data converter)
    return ScannerProperties(ScannerType::Movia, // scannerType
                             100, // nbOfLayers
                             3, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             128, // verticalResolution
                             NaN, // distanceAccuracy
                             NaN, // normTargetRange
                             NaN, // minScanAngle
                             NaN); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyRLidarRSBP)>()
{
    return ScannerProperties(ScannerType::ThirdPartyRLidarRSBP, // scannerType
                             32, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             2.81F * deg2radf, // verticalResolution
                             0.01F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyRLidarRSR32)>()
{
    return ScannerProperties(ScannerType::ThirdPartyRLidarRSR32, // scannerType
                             32, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             0.33F * deg2radf, // verticalResolution +1.66deg ~ -4.66deg
                             0.05F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyRLidarRSR128)>()
{
    return ScannerProperties(ScannerType::ThirdPartyRLidarRSR128, // scannerType
                             128, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             0.1F * deg2radf, // verticalResolution
                             0.03F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyRLidarRSRP128)>()
{
    return ScannerProperties(ScannerType::ThirdPartyRLidarRSRP128, // scannerType
                             128, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             0.1F * deg2radf, // verticalResolution
                             0.02F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyVLidarVV32)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyVLidarVV32, // scannerType
                             32, // nbOfLayers
                             std::numeric_limits<uint8_t>::quiet_NaN(), // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             NaN, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyVLidarVH64)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyVLidarVH64, // scannerType
                             64, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.02F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyVLidarVH32)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyVLidarVH32, // scannerType
                             32, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             1.33F * deg2radf, // verticalResolution
                             0.02F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyVLidarVS128)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyVLidarVS128, // scannerType
                             128, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             1.33F * deg2radf, // verticalResolution
                             0.04F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidar40)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidar40, // scannerType
                             40, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.04F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidar40P)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidar40P, // scannerType
                             40, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.04F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidar64)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidar64, // scannerType
                             64, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.04F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidar128)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidar128, // scannerType
                             128, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.02F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidarQT128)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidarQT128, // scannerType
                             128, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.03F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidarQT64)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidarQT64, // scannerType
                             64, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.03F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidarXT32)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidarXT32, // scannerType
                             32, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.01F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyHLidarXT16)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyHLidarXT16, // scannerType
                             16, // nbOfLayers
                             2, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.01F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================

template<>
inline ScannerProperties
ScannerProperties::createScannerProperties<static_cast<uint8_t>(ScannerType::ThirdPartyOLidar32)>()
{
    // This sensor uses a different vertical spacing for the upper and lower block of lasers -> no common vertical resolution.
    return ScannerProperties(ScannerType::ThirdPartyOLidar32, // scannerType
                             32, // nbOfLayers
                             1, // maxNbOfEchos
                             NaN, // verticalBeamDivergence
                             NaN, // horizontalBeamDivergence
                             NaN, // verticalResolution
                             0.03F, // distanceAccuracy
                             NaN, // normTargetRange
                             -pif, // minScanAngle
                             pif); // maxScanAngle
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
