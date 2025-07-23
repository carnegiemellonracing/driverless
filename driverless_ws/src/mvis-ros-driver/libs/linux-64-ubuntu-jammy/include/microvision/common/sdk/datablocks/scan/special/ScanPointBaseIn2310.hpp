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
//! \date Oct 23, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class MICROVISION_SDK_API ScanPointBaseIn2310
{
public:
    enum class RawFlags : uint16_t
    {
        Transparent = 0x0001,
        Rain        = 0x0002,
        Ground      = 0x0004, // not used
        Dirt        = 0x0008, // always unset

        HighThresholdH1 = 0x0010,
        HighTresholdH1  = HighThresholdH1, //!< Deprecated
        // invalid 0x0020
        NearRange       = 0x0040,
        HighThresholdH2 = 0x0080, // ==> MARKER
        HighTresholdH2  = HighThresholdH2, //!< Deprecated

        Noise            = 0x0100,
        CandidateInvalid = 0x0200, // unused
        RainStep1Done    = 0x0400, // unused
        RainStep2Done    = 0x0800,

        GroundStep1Done     = 0x1000, // unused
        GroundStep2Done     = 0x2000, // unused
        BlueValidCalculated = 0x4000, // unused
        BlueValidCaculated  = BlueValidCalculated, //!< Deprecated
        Flushed             = 0x8000 //!< Scan point has completely preprocessed.
    };

public:
    static std::streamsize getSerializedSize_static() { return 16; }

public:
    ScanPointBaseIn2310();
    virtual ~ScanPointBaseIn2310();

public:
    //! Equality predicate
    bool operator==(const ScanPointBaseIn2310& other) const;
    bool operator!=(const ScanPointBaseIn2310& other) const;

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    virtual uint16_t getBlockId() const = 0;

public:
    uint16_t getRadialDistance() const { return m_radialDistance; }
    uint16_t getEchoPulseWidth() const { return m_echoPulseWidth; }
    int16_t getAngle() const { return m_angle; }
    uint16_t getReserved() const { return m_reserved; }
    uint16_t getResolution() const { return m_resolution; }
    uint8_t getChannelId() const { return m_channelId; }
    uint8_t getEchoId() const { return m_echoId; }
    uint8_t getFlagsHigh() const { return m_flagsHigh; }
    uint8_t getFlagsLow() const { return m_flagsLow; }

public:
    void setRadialDistance(const uint16_t dist) { m_radialDistance = dist; }
    void setEchoPulseWidth(const uint16_t width) { m_echoPulseWidth = width; }
    void setAngle(const int16_t angle) { m_angle = angle; }
    void setResolution(const uint16_t res) { m_resolution = res; }
    void setChannelId(const uint8_t id) { m_channelId = id; }
    void setEchoId(const uint8_t id) { m_echoId = id; }
    void setFlagsHigh(const uint8_t flags) { m_flagsHigh = flags; }
    void setFlagsLow(const uint8_t flags) { m_flagsLow = flags; }

protected:
    uint16_t m_radialDistance;
    uint16_t m_echoPulseWidth;
    int16_t m_angle;
    uint16_t m_reserved;
    uint16_t m_resolution;
    uint8_t m_channelId;
    uint8_t m_echoId;
    uint8_t m_flagsHigh;
    uint8_t m_flagsLow;
}; // ScanPointBaseIn2310

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
