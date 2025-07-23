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

class ScanInfoIn2310 final
{
public:
    static std::streamsize getSerializedSize_static() { return 16; }

public:
    ScanInfoIn2310();
    virtual ~ScanInfoIn2310();

public:
    //! Equality predicate
    bool operator==(const ScanInfoIn2310& other) const;
    bool operator!=(const ScanInfoIn2310& other) const;

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint16_t getApdActualValue() const { return m_ApdActualValue; }
    uint16_t getApdTableValue() const { return m_ApdTableValue; }
    uint16_t getTemperature() const { return m_temperature; }
    uint16_t getReferenceNoise() const { return m_referenceNoise; }
    uint16_t getControlNoise() const { return m_controlNoise; }
    uint16_t getNoiseMeasurementThreshold() const { return m_noiseMeasurementThreshold; }
    uint16_t getReservedInfo7() const { return m_reservedInfo7; }

public:
    void setApdActualValue(const uint16_t apdActual) { m_ApdActualValue = apdActual; }
    void setApdTableValue(const uint16_t apdTable) { m_ApdTableValue = apdTable; }
    void setTemperature(const uint16_t temp) { m_temperature = temp; }
    void setReferenceNoise(const uint16_t refNoise) { m_referenceNoise = refNoise; }
    void setControlNoise(const uint16_t controlNoise) { m_controlNoise = controlNoise; }
    void setNoiseMeasurementThreshold(const uint16_t threshold) { m_noiseMeasurementThreshold = threshold; }

public:
    static const uint16_t blockId;

protected:
    uint16_t m_ApdActualValue;
    uint16_t m_ApdTableValue;
    uint16_t m_temperature;
    uint16_t m_referenceNoise;
    uint16_t m_controlNoise;
    uint16_t m_noiseMeasurementThreshold;
    uint16_t m_reservedInfo7;
}; // ScanInfoIn2310

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
