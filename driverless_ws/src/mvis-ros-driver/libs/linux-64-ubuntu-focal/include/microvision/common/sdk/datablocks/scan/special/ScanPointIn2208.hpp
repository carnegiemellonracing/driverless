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
//! \date Aug 2, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <istream>
#include <ostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Raw scan point for Scala and Minilux
//------------------------------------------------------------------------------
class ScanPointIn2208 final
{
public:
    enum class Flags : uint16_t
    {
        Transparent = 0x0001U, //!< A point classified as transparent.
        Rain        = 0x0002U, //!< A point classified as rain, snow, spray or fog.
        Ground      = 0x0004U, //!< A point classified as ground.
        Dirt        = 0x0008U, //!< A point classified as dirt on the front window.

        HighThreshold = 0x0010U, //!< For short range measurements: the higher threshold is used.
        Noise         = 0x0020U, //!< A point identified as noise.
        NearRange     = 0x0040U, //!< This bit tags if a point is measured in short range.
        Marker        = 0x0080U, //!< Application depended flag.

        Blooming   = 0x0100U, //!< A point identified as result of blooming effect (previously: right covered).
        Background = 0x0200U, //!< A point classified as background.
        Reserved3  = 0x0400U, //!< Reserved
        Reserved4  = 0x0800U, //!< Reserved

        Reflector     = 0x1000U, //!< A point classified as reflected.
        Reserved5     = 0x2000U, //!< Reserved
        InterlacingPt = 0x4000U, //!< Interlacing Point. Only for A1 prototype.
        Reserved7     = 0x8000U //!< Reserved
    };

public:
    ScanPointIn2208();
    virtual ~ScanPointIn2208();

public:
    //! Equality predicate
    bool operator==(const ScanPointIn2208& other) const;

    bool operator!=(const ScanPointIn2208& other) const;

public:
    static std::streamsize getSerializedSize_static() { return 11; }

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint8_t getThresholdId() const { return m_thresholdId; }
    uint8_t getEchoId() const { return m_echoId; }
    uint8_t getReserved() const { return m_reserved; }
    uint8_t getLayerId() const { return m_layerId; }
    uint16_t getFlags() const { return m_flags; }
    int16_t getHorizontalAngle() const { return m_horizontalAngle; }
    uint16_t getRadialDistance() const { return m_radialDistance; }
    uint16_t getEchoPulseWidth() const { return m_echoPulseWidth; }
    uint8_t getPfValue() const { return m_pfValue; }

public:
    void setThresholdId(const uint8_t newThresholdId) { this->m_thresholdId = newThresholdId & 0x0f; } // 4 bit
    void setEchoId(const uint8_t newEchoId) { this->m_echoId = newEchoId & 0x03; } // 2 bit
    void setReserved(const uint8_t newReserved) { this->m_reserved = newReserved & 0x03; } // 2 bit
    void setLayerId(const uint8_t newLayerId) { this->m_layerId = newLayerId; }
    void setFlags(const uint16_t newFlags) { this->m_flags = newFlags; }
    void setHorizontalAngle(const int16_t newHorizontalAngle) { this->m_horizontalAngle = newHorizontalAngle; }
    void setRadialDistance(const uint16_t newRadialDistance) { this->m_radialDistance = newRadialDistance; }
    void setEchoPulseWidth(const uint16_t newEchoPulseWidth) { this->m_echoPulseWidth = newEchoPulseWidth; }
    void setPfValue(const uint8_t newPfValue) { this->m_pfValue = newPfValue; }

public:
    //	static const uint8_t undefinedThresholdId = 0xF; // 0xE
protected:
    //! \breif  The threshold id. Only 4 bits are used.
    //!
    //! bit 0: gain that has been used - low/high gain 0/1.
    //! bits 1..3: id of the used threshold, values 0 .. 6
    //!
    //! Scala:
    //! 0 = Low threshold,
    //! 1= First High Threshold (H1),
    //! 2 = Second High Threshold (H2)
    //!
    //! Invalid values: 0x7,0xE,0xF
    uint8_t m_thresholdId;
    uint8_t m_echoId; //!<  The echo id. Only 2 bits are used.
    uint8_t m_reserved; //!< Reserved. Only 2 bits are used.

    uint8_t m_layerId; //!< Id of the layer.
    uint16_t m_flags; //!< Point processing flags.
    int16_t m_horizontalAngle; //!< Measurement angle in AngleTicks. [ticks]
    uint16_t m_radialDistance; //!< Scan point distance in cm in scanner coordinate system. [cm]
    uint16_t m_echoPulseWidth; //!< Echo pulse width in cm. [cm]
    uint8_t m_pfValue; //!< False probability of this measurement (range: 0 - 100; invalid value: 0xFFU)
}; // ScanPointIn2208

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
