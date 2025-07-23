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
//! \date Sep 17, 2015
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

class ScanHeaderIn2310 final
{
public:
    static std::streamsize getSerializedSize_static() { return 16; }

public:
    ScanHeaderIn2310();
    virtual ~ScanHeaderIn2310();

public:
    //! Equality predicate
    bool operator==(const ScanHeaderIn2310& other) const;

    bool operator!=(const ScanHeaderIn2310& other) const;

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint16_t getScanCounter() const { return m_scanCounter; }
    uint16_t getMinApdOffset() const { return m_minApdOffset; }
    uint16_t getMaxApdOffset() const { return m_maxApdOffset; }
    uint16_t getFrequencyInteger() const { return m_frequencyInteger; }
    uint16_t getFrequencyFractional() const { return m_frequencyFractional; }
    uint16_t getDeviceId() const { return m_deviceId; }
    uint16_t getReservedHeader7() const { return m_reservedHeader7; }

public:
    void setScanCounter(const uint16_t scanCounter) { m_scanCounter = scanCounter; }
    void setMinApdOffset(const uint16_t minApdOffset) { m_minApdOffset = minApdOffset; }
    void setMaxApdOffset(const uint16_t maxApdOffset) { m_maxApdOffset = maxApdOffset; }
    void setFrequencyInteger(const uint16_t freqInteger) { m_frequencyInteger = freqInteger; }
    void setFrequencyFractional(const uint16_t freqFrac) { m_frequencyFractional = freqFrac; }
    void setDeviceId(const uint16_t deviceId) { m_deviceId = deviceId; }

public:
    static const uint16_t blockId;

protected:
    uint16_t m_scanCounter;
    uint16_t m_minApdOffset;
    uint16_t m_maxApdOffset;
    uint16_t m_frequencyInteger;
    uint16_t m_frequencyFractional;
    uint16_t m_deviceId;
    uint16_t m_reservedHeader7;
}; // ScanHeaderIn2310

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
