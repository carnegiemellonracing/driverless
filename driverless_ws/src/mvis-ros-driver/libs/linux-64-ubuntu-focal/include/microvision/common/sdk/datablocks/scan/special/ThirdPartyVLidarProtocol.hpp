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
//! \date Jun 08, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <array>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class defining constants and enums according to the ThirdPartyVLidar network protocol.
//------------------------------------------------------------------------------
class ThirdPartyVLidarProtocol final
{
public:
    // Protocol constants.
    static constexpr uint16_t upperBlockId{
        0xEEFFU}; //!< ID of a block of lasers for the VH32 or the upper 32 lasers of the VH64
    static constexpr uint16_t lowerBlockId{0xDDFFU}; //!< ID of a block of lasers for the lower 32 lasers of the VH64

    static constexpr uint32_t lasersPerBlock{32}; //!< no. of laser data per block
    static constexpr uint32_t blocksPerPacket{12}; //!< no. of firing data blocks per UDP packet

    //! value for strongest echo only return mode as defined in factory bytes (VH32, VV16 and VV32C only)
    static constexpr uint8_t returnModeValueStrongest{0x37U};
    //! value for last echo only return mode as defined in factory bytes (VH32, VV16 and VV32C only)
    static constexpr uint8_t returnModeValueLast{0x38U};
    //! value for dual return mode as defined in factory bytes (VH32 only, VV16 and VV32C)
    static constexpr uint8_t returnModeValueDual{0x39U};
    //! product ID of VH32E as defined in factory bytes (VS128, VH32, VV16 and VV32C only)
    static constexpr uint8_t productIdValueVH32E{0x21U};
    //! product ID of VV16 as defined in factory bytes (VS128, VH32, VV16 and VV32C only)
    static constexpr uint8_t productIdValueVV16{0x22U};
    //! product ID of Puck LITE as defined in factory bytes (VS128, VH32, VV16 and VV32C only)
    static constexpr uint8_t productIdValuePuckLite{0x22U};
    //! product ID of Puck Hi-Res as defined in factory bytes (VS128, VH32, VV16 and VV32C only)
    static constexpr uint8_t productIdValuePuckHiRes{0x24U};
    //! product ID of VV32C as defined in factory bytes (VS128, VH32, VV16 and VV32C only)
    static constexpr uint8_t productIdValueVV32C{0x28U};
    //! product ID of Velarray as defined in factory bytes (VS128, VH32, VV16 and VV32C only)
    static constexpr uint8_t productIdValueVelarray{0x31U};
    //! product ID of VS128 as defined in factory bytes (VS128, VH32, VV16 and VV32C only)
    static constexpr uint8_t productIdValueVS128{0xA1U};

public:
    static constexpr uint32_t firingDataContainerSize{100}; //!< Size of firing data container.
    static constexpr uint32_t udpPayloadSize{1206}; //!< UDP payload size of a ThirdPartyVLidar Lidar packet.

public:
    //==============================================================================
    //! \brief The type of ThirdPartyVLidar device.
    //------------------------------------------------------------------------------
    enum class DeviceType : uint8_t
    {
        Unknown = 0,
        VH64    = 1,
        VH32    = 2,
        VV16    = 3,
        VV32    = 4,
        VS128   = 5
    }; // DeviceType

    //==============================================================================
    //! \brief The scanner return mode, i.e. whether the scanner returns one (single) or two (dual) points per laser.
    //------------------------------------------------------------------------------
    enum class ReturnMode : uint8_t
    {
        Unknown = 0, //!< Return mode not known yet.
        Single  = 1, //!< Either strongest or last signal is returned (which of them does not matter).
        Dual    = 2 //!< Both signals - strongest and last - are returned.
    }; // ReturnMode

public:
    //========================================
    //! \brief Get the timestamp of a laser data block in a UDP data packet sent from a ThirdPartyVLidar device.
    //!
    //! \param[in] deviceType  Type of ThirdPartyVLidar device.
    //! \param[in] returnMode  Return mode of the scanner.
    //! \param[in] blockIdx    Index of the block to get the timestamp offset for.
    //! \return offset of the given laser data block to the packet timestamp in nanoseconds.
    //----------------------------------------
    static int32_t
    getBlockTimestampOffset(const DeviceType deviceType, const ReturnMode returnMode, const uint32_t blockIdx);

    //========================================
    //! \brief Get the timestamp of an individual laser in a data block of a UDP data packet
    //!        sent from a ThirdPartyVLidar device.
    //!
    //! \param[in] deviceType  Type of ThirdPartyVLidar device.
    //! \param[in] returnMode  Return mode of the scanner.
    //! \param[in] laserIdx    Index of the laser to get the timestamp offset for.
    //! \return offset of the laser to the laser data block timestamp in nanoseconds.
    //----------------------------------------
    static uint32_t
    getLaserTimestampOffset(const DeviceType deviceType, const ReturnMode returnMode, const uint32_t laserIdx);

    //========================================
    //! \brief Get the minimum timestamp of a ThirdPartyVLidar device.
    //!
    //! \param[in] deviceType  Type of ThirdPartyVLidar device.
    //! \param[in] returnMode  Return mode of the scanner.
    //! \return minimum timestamp offset in nanoseconds.
    //----------------------------------------
    static int32_t getMinimumTimestampOffset(const DeviceType deviceType, const ReturnMode returnMode);

    //========================================
    //! \brief Get the maximum time between two laser firings of a ThirdPartyVLidar device.
    //!
    //! \param[in] deviceType  Type of ThirdPartyVLidar device.
    //! \param[in] returnMode  Return mode of the scanner.
    //! \return maximum time between two laser firings in nanoseconds.
    //----------------------------------------
    static uint32_t getMinimumLaserFiringTime(const DeviceType deviceType, const ReturnMode returnMode);

private:
    //========================================
    //! \brief Constructor (no instances allowed).
    //----------------------------------------
    ThirdPartyVLidarProtocol() = delete;

private:
    //==============================================================================
    // The following tables were created using the Python script
    // 'ThirdPartyVLidarTimeAdjust.py'.
    // --- Do not modify! ---
    //==============================================================================
    static const std::array<int32_t, 12> blockTimeOffsetsVH64SingleReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVH64SingleReturn;
    static const std::array<int32_t, 12> blockTimeOffsetsVH64DualReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVH64DualReturn;
    static const std::array<int32_t, 12> blockTimeOffsetsVH32SingleReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVH32SingleReturn;
    static const std::array<int32_t, 12> blockTimeOffsetsVH32DualReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVH32DualReturn;
    static const std::array<int32_t, 12> blockTimeOffsetsVV16SingleReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVV16SingleReturn;
    static const std::array<int32_t, 12> blockTimeOffsetsVV16DualReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVV16DualReturn;
    static const std::array<int32_t, 12> blockTimeOffsetsVV32SingleReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVV32SingleReturn;
    static const std::array<int32_t, 12> blockTimeOffsetsVV32DualReturn;
    static const std::array<int32_t, 2> blockTimeOffsetRangeVV32DualReturn;

    static const std::array<uint32_t, 32> laserTimeOffsetsVH64SingleReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVH64SingleReturn;
    static const uint32_t minLaserTimeVH64SingleReturn;
    static const std::array<uint32_t, 32> laserTimeOffsetsVH64DualReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVH64DualReturn;
    static const uint32_t minLaserTimeVH64DualReturn;
    static const std::array<uint32_t, 32> laserTimeOffsetsVH32SingleReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVH32SingleReturn;
    static const uint32_t minLaserTimeVH32SingleReturn;
    static const std::array<uint32_t, 32> laserTimeOffsetsVH32DualReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVH32DualReturn;
    static const uint32_t minLaserTimeVH32DualReturn;
    static const std::array<uint32_t, 32> laserTimeOffsetsVV16SingleReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVV16SingleReturn;
    static const uint32_t minLaserTimeVV16SingleReturn;
    static const std::array<uint32_t, 32> laserTimeOffsetsVV16DualReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVV16DualReturn;
    static const uint32_t minLaserTimeVV16DualReturn;
    static const std::array<uint32_t, 32> laserTimeOffsetsVV32SingleReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVV32SingleReturn;
    static const uint32_t minLaserTimeVV32SingleReturn;
    static const std::array<uint32_t, 32> laserTimeOffsetsVV32DualReturn;
    static const std::array<uint32_t, 2> laserTimeOffsetRangeVV32DualReturn;
    static const uint32_t minLaserTimeVV32DualReturn;

    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVH64SingleReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVH64SingleReturn;
    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVH64DualReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVH64DualReturn;
    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVH32SingleReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVH32SingleReturn;
    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVH32DualReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVH32DualReturn;
    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVV16SingleReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVV16SingleReturn;
    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVV16DualReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVV16DualReturn;
    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVV32SingleReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVV32SingleReturn;
    static const std::array<std::array<int32_t, 32>, 12> timeOffsetsVV32DualReturn;
    static const std::array<int32_t, 2> timeOffsetRangeVV32DualReturn;
}; // ThirdPartyVLidarProtocol

//==============================================================================

inline std::ostream& operator<<(std::ostream& os, ThirdPartyVLidarProtocol::DeviceType deviceType)
{
    switch (deviceType)
    {
    case ThirdPartyVLidarProtocol::DeviceType::Unknown:
        os << "<unknown>";
        break;

    case ThirdPartyVLidarProtocol::DeviceType::VH64:
        os << "VH64";
        break;

    case ThirdPartyVLidarProtocol::DeviceType::VH32:
        os << "VH32";
        break;

    case ThirdPartyVLidarProtocol::DeviceType::VV16:
        os << "VV16";
        break;

    case ThirdPartyVLidarProtocol::DeviceType::VV32:
        os << "VV32";
        break;

    case ThirdPartyVLidarProtocol::DeviceType::VS128:
        os << "VS128";
        break;

    default:
        throw std::invalid_argument("Unknown device type!");
    }

    return os;
}

//==============================================================================

inline std::ostream& operator<<(std::ostream& os, ThirdPartyVLidarProtocol::ReturnMode returnMode)
{
    switch (returnMode)
    {
    case ThirdPartyVLidarProtocol::ReturnMode::Unknown:
        os << "<unknown>";
        break;

    case ThirdPartyVLidarProtocol::ReturnMode::Single:
        os << "Single";
        break;

    case ThirdPartyVLidarProtocol::ReturnMode::Dual:
        os << "Dual";
        break;

    default:
        throw std::invalid_argument("Unknown device type!");
    }

    return os;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
