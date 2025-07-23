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

class ScanSegInfoIn2310 final
{
public:
    static std::streamsize getSerializedSize_static() { return 16; }

public:
    ScanSegInfoIn2310();
    virtual ~ScanSegInfoIn2310();

public:
    //! Equality predicate
    bool operator==(const ScanSegInfoIn2310& other) const;
    bool operator!=(const ScanSegInfoIn2310& other) const;

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint16_t getBlockId() const { return blockId; }

public:
    uint16_t getSegmentIndex() const { return m_segmentIndex; }
    uint16_t getApdVoltage() const { return m_ApdVoltage; }
    uint16_t getNoise() const { return m_noise; }
    uint16_t getReservedSegmentInfo4() const { return m_reservedSegmentInfo4; }
    uint16_t getReservedSegmentInfo5() const { return m_reservedSegmentInfo4; }
    uint16_t getReservedSegmentInfo6() const { return m_reservedSegmentInfo4; }
    uint16_t getReservedSegmentInfo7() const { return m_reservedSegmentInfo4; }

public:
    void setSegmentIndex(const uint16_t idx) { m_segmentIndex = idx; }
    void setApdVoltage(const uint16_t voltage) { m_ApdVoltage = voltage; }
    void setNoise(const uint16_t noise) { m_noise = noise; }

public:
    static const uint16_t blockId;

protected:
    uint16_t m_segmentIndex;
    uint16_t m_ApdVoltage;
    uint16_t m_noise;
    uint16_t m_reservedSegmentInfo4;
    uint16_t m_reservedSegmentInfo5;
    uint16_t m_reservedSegmentInfo6;
    uint16_t m_reservedSegmentInfo7;
}; // ScanSegInfoIn2310

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
