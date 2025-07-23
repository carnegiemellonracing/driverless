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
//! \date Dec 18, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <sstream>
#include <string>
#include <cstdint>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Ids of some ScannerTypes
//------------------------------------------------------------------------------
enum class ScannerType : uint8_t
{
    Invalid = 0xFFU, //!< invalid value ( = -1)
    // 0x00U reserved
    // 0x01U reserved
    // 0x02U reserved
    // 0x03U reserved
    Ecu = 0x04U, //!< MVIS ECU
    // 0x05U reserved
    Lux = 0x06U, //!< MVIS LUX3
    // 0x07U reserved
    // 0x08U reserved
    Lux4 = 0x10U, //!< MVIS LUX4
    // 0x11U reserved
    // 0x18U reserved
    // 0x19U reserved
    // 0x20U reserved
    // 0x21U reserved
    // 0x28U reserved
    // 0x29U reserved
    MiniLux = 0x30U, //!< MVIS MiniLux
    LuxHr   = 0x31U, //!< MVIS LUX High resolution scanner

    // 0x40U reserved
    // 0x41U reserved
    // 0x42U reserved

    ThirdPartySLidarL200  = 0x50U, //!< Third party lidar sensor 200/291
    ThirdPartySLidarL100  = 0x51U, //!< Third party lidar sensor 100/111
    ThirdPartySLidarJeff  = 0x52U, //!< Third party lidar sensor
    ThirdPartySLidarT300  = 0x53U, //!< Third party lidar sensor 300
    ThirdPartySLidarL500  = 0x54U, //!< Third party lidar sensor 500/511
    ThirdPartySLidarM1000 = 0x55U, //!< Third party lidar sensor
    ThirdPartySLidarL1000 = 0x56U, //!< Third party lidar sensor

    // 0x60U reserved
    ScalaB2 = 0x62U, //!< ScalaB2
    // 0x63U reserved
    // 0x6DU reserved
    // 0x6EU reserved
    // 0x6FU reserved

    Movia = 0x70U, //!< MOVIA Lidar with LDMIv2
    // 0x71U reserved
    // 0x72U reserved
    // 0x73U reserved
    // 0x74U reserved

    // 0x90U reserved

    Mavin = 0xA0, //! Mavin

    ThirdPartyRLidarRSBP    = 0xB0U, //!< Third party lidar sensor with 32 channels, short range, wide FOV.
    ThirdPartyRLidarRSR32   = 0xB1U, //!< Third party lidar sensor with 32 channels.
    ThirdPartyRLidarRSR128  = 0xB2U, //!< Third party lidar sensor with 128 channels.
    ThirdPartyRLidarRSRP128 = 0xB3U, //!< Third party lidar sensor with 128 channels.

    ThirdPartyVLidarVV16  = 0xC8U, //!< Third party lidar sensor with 16 channels.
    ThirdPartyVLidarVV32  = 0xC9U, //!< Third party lidar sensor with 32 channels.
    ThirdPartyVLidarVH64  = 0xD2U, //!< Third party lidar sensor with 64 channels.
    ThirdPartyVLidarVH32  = 0xD3U, //!< Third party lidar sensor with 32 channels.
    ThirdPartyVLidarVS128 = 0xD4U, //!< Third party lidar sensor with 128 channels.

    ThirdPartyHLidar40    = 0xE0U, //!< Third party lidar sensor with 40 channels.
    ThirdPartyHLidar40P   = 0xE1U, //!< Third party lidar sensor with 40 channels.
    ThirdPartyHLidar64    = 0xE2U, //!< Third party lidar sensor with 64 channels.
    ThirdPartyHLidar128   = 0xE3U, //!< Third party lidar sensor with 128 channels.
    ThirdPartyHLidarQT128 = 0xE4U, //!< Third party lidar sensor with 128 channels (QT Version).
    ThirdPartyHLidarQT64  = 0xE5U, //!< Third party lidar sensor with 64 channels 64 channels (QT Version).
    ThirdPartyHLidarXT32  = 0xE6U, //!< Third party lidar sensor with 32 channels (XT Version).
    ThirdPartyHLidarXT16  = 0xE7U, //!< Third party lidar sensor with 16 channels 128 channels (XT Version).

    ThirdPartyOLidar32  = 0xF0U, //!< Third party lidar sensor with 32 channels.
    ThirdPartyOLidar64  = 0xF1U, //!< Third party lidar sensor with 64 channels.
    ThirdPartyOLidar128 = 0xF2U, //!< Third party lidar sensor with 128 channels.

    // 0x0101U reserved
    // 0x0102U reserved
    // 0x0700U reserved

}; // ScannerType
//==============================================================================

//==============================================================================
//! \brief Create \sa ScannerType from a given string.
//! \param[in] string             Input to be transformed into \sa ScannerType
//! \return Type of \sa ScannerType. Returns \sa ScannerType::Invalid if casting failed.
//------------------------------------------------------------------------------
ScannerType makeScannerType(const std::string& scannerType);

//! \brief Convert ScannerType to string.
//! \param[in] scannerType    The scannerType to be converted
//! \return String with representation of \sa ScannerType
//------------------------------------------------------------------------------
std::string to_string(ScannerType val);

//! \brief Stream operator for writing the scannerType to a stream
//! \param[in,out] out             The stream, the scannerType shall be written to
//! \param[in] scannerType    The scannerType which shall be streamed
//! \return The stream to which the scannerType was written to
//------------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& out, const ScannerType& scannerType)
{
    out << to_string(scannerType);
    return out;
}

//==============================================================================
//! \brief Stream operator for reading ScannerType from a stream
//! \param[in] is             The stream from which the scannerType shall read
//! \param[in] scannerType    The scannerType  which shall be filled.
//! \return The stream from which the scannerType is read.
//!
//! \note If reading the data failed (check with /a istream::fail()) the content of the ScannerType is undefined.
//------------------------------------------------------------------------------
inline std::istream& operator>>(std::istream& is, ScannerType& scannerType)
{
    char buffer[32] = {0}; //max-length := 32 char
    is.get(buffer, sizeof(buffer));
    if (is.fail())
    {
        // Error, give up.
        return is;
    }
    scannerType = makeScannerType((buffer));

    if (scannerType == ScannerType::Invalid)
    {
        // Unknown string -> error.
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================