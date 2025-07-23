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
//! \date Jan 19, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/Version448In6301.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/SerialNumberIn6301.hpp>
#include <microvision/common/sdk/datablocks/scan/ScannerInfo.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Device Status of LUX3 and Scala B2 X90
//!
//! General data type: \ref microvision::common::sdk::DeviceStatus
//------------------------------------------------------------------------------
class DeviceStatus6301 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.devicestatus6301"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    constexpr static const uint8_t nbOfApdSectors{10};
    constexpr static const uint8_t nbOfReservedA{21};

public:
    using AdaptiveApdVoltageArray = std::array<float, nbOfApdSectors>;
    using ActualNoiseArray        = std::array<uint16_t, nbOfApdSectors>;
    using ReservedArray           = std::array<uint16_t, nbOfReservedA>;

public:
    DeviceStatus6301();
    virtual ~DeviceStatus6301() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

    const SerialNumberIn6301& getSerialNumber() const { return m_serialNumberOfScanner; }
    SerialNumberIn6301& getSerialNumber() { return m_serialNumberOfScanner; }

    ScannerType getScannerType() const { return m_scannerType; }
    uint8_t getReserved0() const { return m_reserved0; }

    const Version448In6301& getFpgaVersion() const { return m_fpgaVersion; }
    Version448In6301& getFpgaVersion() { return m_fpgaVersion; }
    const Version448In6301& getDspVersion() const { return m_dspVersion; }
    Version448In6301& getDspVersion() { return m_dspVersion; }
    const Version448In6301& getHostVersion() const { return m_hostVersion; }
    Version448In6301& getHostVersion() { return m_hostVersion; }

    uint16_t getFpgaModusRegister() const { return m_fpgaModusRegister; }
    uint16_t getReserved1() const { return m_reserved1; }
    float getSensorTemperature() const { return m_sensorTemperature; }
    float getFrequency() const { return m_frequency; }
    float getApdTableVoltage() const { return m_apdTableVoltage; }

    float getAdaptiveApdVoltage(const uint8_t idx) const { return m_adaptiveApdVoltage.at(idx); }
    const AdaptiveApdVoltageArray& getAdaptiveApdVoltageArray() const { return m_adaptiveApdVoltage; }
    float getMinApdVoltageOffset() const { return m_minApdVoltageOffset; }
    float getMaxApdVoltageOffset() const { return m_maxApdVoltageOffset; }
    uint16_t getReserved2() const { return m_reserved2; }
    uint16_t getReserved3() const { return m_reserved3; }
    float getNoiseMeasurementThreshold() const { return m_noiseMeasurementThreshold; }
    uint16_t getReferenceNoise() const { return m_referenceNoise; }

    uint16_t getActualNoise(const uint8_t idx) const { return m_actualNoise.at(idx); }
    const ActualNoiseArray& getActualNoiseArray() const { return m_actualNoise; }

    uint16_t getReservedA(const uint8_t idx) const { return m_reservedA.at(idx); }
    const ReservedArray& getReservedAArray() const { return m_reservedA; }

public:
    void setScannerType(const ScannerType newScannerType) { m_scannerType = newScannerType; }
    //	void setReserved0(const uint8_t newReserved0) { m_reserved0 = newReserved0; }
    void setFpgaModusRegister(const uint16_t newFpgaModusRegister) { m_fpgaModusRegister = newFpgaModusRegister; }
    //	void setReserved1(const uint16_t newReserved1) { m_reserved1 = newReserved1; }
    void setSensorTemperature(const float newSensorTemperature) { m_sensorTemperature = newSensorTemperature; }
    void setFrequency(const float newFrequency) { m_frequency = newFrequency; }
    void setApdTableVoltage(const float newApdTableVoltage) { m_apdTableVoltage = newApdTableVoltage; }
    void setAdaptiveApdVoltage(const uint32_t idx, const float newAdaptiveApdVoltage)
    {
        m_adaptiveApdVoltage[idx] = newAdaptiveApdVoltage;
    }
    void setMinApdVoltageOffset(const float newMinApdVoltageOffset) { m_minApdVoltageOffset = newMinApdVoltageOffset; }
    void setMaxApdVoltageOffset(const float newMaxApdVoltageOffset) { m_maxApdVoltageOffset = newMaxApdVoltageOffset; }
    //	void setReserved2(const uint16_t newReserved2) { m_reserved2 = newReserved2; }
    //	void setReserved3(const uint16_t newReserved3) { m_reserved3 = newReserved3; }
    void setNoiseMeasurementThreshold(const float newNoiseMeasurementThreshold)
    {
        m_noiseMeasurementThreshold = newNoiseMeasurementThreshold;
    }
    void setReferenceNoise(const uint16_t newReferenceNoise) { m_referenceNoise = newReferenceNoise; }
    void setActualNoise(const uint32_t idx, const uint16_t newActualNoise) { m_actualNoise[idx] = newActualNoise; }
    //	void setReservedA(const int idx, const uint16_t newReserved) { m_reservedA[idx] = newReserved; }

protected:
    SerialNumberIn6301 m_serialNumberOfScanner{}; //!< Serial number of the scanner.
    ScannerType m_scannerType{}; //!< Type of the scanner.
    uint8_t m_reserved0{0};

    Version448In6301 m_fpgaVersion{}; //!< Version of the FPGA.
    Version448In6301 m_dspVersion{}; //!< Version of the DSP.
    Version448In6301 m_hostVersion{}; //!< Version of the host.

    uint16_t m_fpgaModusRegister{}; //!< State of the FPGA modus register.
    uint16_t m_reserved1{};
    float m_sensorTemperature{0.0F}; //!< Sensor temperature in °C.
    float m_frequency{0.0F}; //!< Sensor APD temperature 1 in °C.
    float m_apdTableVoltage{0.0F};
    AdaptiveApdVoltageArray m_adaptiveApdVoltage;
    float m_minApdVoltageOffset{0.0F}; //!< Minimal APD voltage offset.
    float m_maxApdVoltageOffset{0.0F}; //!< Maximal APD voltage offset.

    uint16_t m_reserved2{0};
    uint16_t m_reserved3{0};

    float m_noiseMeasurementThreshold{0.0F}; //!< Noise measurement threshold.
    uint16_t m_referenceNoise{0}; //!< Reference noise.
    ActualNoiseArray m_actualNoise;
    ReservedArray m_reservedA;
}; // DeviceStatus6301

//==============================================================================

//==============================================================================

bool operator==(const DeviceStatus6301& lhs, const DeviceStatus6301& rhs);
bool operator!=(const DeviceStatus6301& lhs, const DeviceStatus6301& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
