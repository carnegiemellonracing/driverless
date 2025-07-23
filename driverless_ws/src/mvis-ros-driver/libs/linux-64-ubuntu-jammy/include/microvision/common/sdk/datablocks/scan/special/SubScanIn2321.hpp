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
//! \date Jun 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2321.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ThirdPartyVLidarProtocol.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <limits>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class used to group a number of raw ThirdPartyVLidar scan points to a sub-scan.
//!
//! The third party lidar scanner network protocol defines a Firing Data Block that contains the echos of 32 lasers at a rotational
//! position of the scanner (scan column). This class is used to represented this data block.
//------------------------------------------------------------------------------
class SubScanIn2321 final
{
    friend bool operator==(const SubScanIn2321& lhs, const SubScanIn2321& rhs);

public:
    //========================================
    //! \brief Type of echo returned in the scan points.
    //----------------------------------------
    enum class EchoType : uint8_t
    {
        Unknown = 0, //!< Type not known (yet).
        Single  = 1, //!< Scan points were received in single return mode (not known whether this was strongest
        //!< or latest echo).
        DualStrongest = 2, //!< Scan points were received in dual return mode; this sub-scan contains the strongest
        //!< echo.
        DualLatest = 3 //!< Scan points were received in dual return mode; this sub-scan contains the latest echo.
    };

    using ScanPointVector = std::vector<ScanPointIn2321>;

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
    SubScanIn2321() = default;

    //========================================
    //! \brief Copy-constructor.
    //!
    //! \param[in] src  object to copy.
    //----------------------------------------
    SubScanIn2321(const SubScanIn2321& src) = default;

    //========================================
    //! \brief Assignment operator.
    //!
    //! \param[in] src  object to be assigned
    //! \return this
    //----------------------------------------
    SubScanIn2321& operator=(const SubScanIn2321& src) = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~SubScanIn2321() = default;

public:
    std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public: // getter
    //========================================
    //! \brief Get the offset of this sub-scan to the scan timestamp-
    //!
    //! \return timestamp offset when the first laser of this sub-scan was fired in nanoseconds.
    //----------------------------------------
    uint32_t getTimestampOffset() const { return m_timestampOffset; }

    //========================================
    //! \brief Get the type of the echos in this sub-scan.
    //!
    //! \return the echo type.
    //----------------------------------------
    EchoType getEchoType() const { return m_echoType; }

    //========================================
    //! \brief Get the block ID as defined in the sensors network protocol.
    //!
    //! \return the block ID as used in the Firing Data Block.
    //!
    //! \note The block ID is used to differentiate between to upper and lower 32 lasers of a VH64.
    //----------------------------------------
    uint16_t getBlockId() const { return m_blockId; }

    //========================================
    //! \brief Get the rotation position of the scanner when this sub-scan was recorded.
    //!
    //! \return rotational position in centi degrees.
    //----------------------------------------
    uint16_t getRotationalPosition() const { return m_rotationalPosition; }

    //========================================
    //! \brief Get the scan point as received from the scanner in the Firing Data Block.
    //!
    //! \return scan points.
    //----------------------------------------
    const ScanPointVector& getScanPoints() const { return m_scanPoints; }

public: // setter
    //========================================
    //! \brief Set the offset of this sub-scan to the scan timestamp-
    //!
    //! \param[in] timestampOffset  new timestamp offset in nanoseconds.
    //----------------------------------------
    void setTimestampOffset(const uint32_t timestampOffset) { m_timestampOffset = timestampOffset; }

    //========================================
    //! \brief Set the type of the echos in this sub-scan.
    //!
    //! \param[in] echoType  the new echo type.
    //----------------------------------------
    void setEchoType(const EchoType echoType) { m_echoType = echoType; }

    //========================================
    //! \brief Set the block ID as defined in the sensors network protocol.
    //!
    //! \param[in] blockId  the new block ID.
    //----------------------------------------
    void setBlockId(const uint16_t blockId) { m_blockId = blockId; }

    //========================================
    //! \brief Set the rotation position of the scanner when this sub-scan was recorded.
    //!
    //! \param[in] rotationalPosition the new rotational position in centi degrees.
    //----------------------------------------
    void setRotationalPosition(const uint16_t rotationalPosition) { m_rotationalPosition = rotationalPosition; }

    //========================================
    //! \brief Set the scan point as received from the scanner in the Firing Data Block.
    //!
    //! \param[in] scanPoints  the new scan points.
    //----------------------------------------
    void setScanPoints(const ScanPointVector& scanPoints);

private:
    uint32_t m_timestampOffset{0};
    // Offset to scan timestamp in nanoseconds for the laser that was fired first.
    EchoType m_echoType{EchoType::Unknown};
    uint16_t m_blockId{0};
    uint16_t m_rotationalPosition{std::numeric_limits<uint16_t>::max()};
    ScanPointVector m_scanPoints;
}; // SubScanIn2321

//==============================================================================

//==============================================================================
//! \brief Test sub-scans for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise
//------------------------------------------------------------------------------
bool operator==(const SubScanIn2321& lhs, const SubScanIn2321& rhs);

//==============================================================================
//! \brief Test sub-scans for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise
//------------------------------------------------------------------------------
inline bool operator!=(const SubScanIn2321& lhs, const SubScanIn2321& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
