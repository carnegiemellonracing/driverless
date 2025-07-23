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
//! \date Jan 29, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/Version448.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/SerialNumber.hpp>
#include <microvision/common/sdk/datablocks/scan/ScannerInfo.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief status of device
//!
//! Special data types:
//! \ref microvision::common::sdk::DeviceStatus6301
//! \ref microvision::common::sdk::DeviceStatus6303
//------------------------------------------------------------------------------
class DeviceStatus final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.devicestatus"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    constexpr static const uint8_t nbOfApdSectors{10};

public:
    using AdaptiveApdVoltageArray = std::array<float, nbOfApdSectors>;
    using ActualNoiseArray        = std::array<uint16_t, nbOfApdSectors>;

public:
    DeviceStatus();
    virtual ~DeviceStatus() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    const SerialNumber& getSerialNumberOfScanner() const { return m_serialNumberOfScanner; }
    ScannerType getScannerType() const { return m_scannerType; }
    uint8_t getDeviceId() const { return m_deviceId; }
    const Version448& getFpgaVersion() const { return m_fpgaVersion; }
    const Version448& getDspVersion() const { return m_dspVersion; }
    const Version448& getHostVersion() const { return m_hostVersion; }
    uint16_t getFpgaOperationRegister() const { return m_fpgaOperationRegister; }
    uint16_t getFpgaStatusRegister() const { return m_fpgaStatusRegister; }
    float getSensorTemperatureApd0() const { return m_sensorTemperatureApd0; }
    float getSensorTemperatureApd1() const { return m_sensorTemperatureApd1; }
    float getScanFrequency() const { return m_scanFrequency; }
    float getApdTableBiasVoltage() const { return m_apdTableBiasVoltage; }
    const AdaptiveApdVoltageArray& getAdaptiveApdVoltageArray() const { return m_adaptiveApdVoltageArray; }
    AdaptiveApdVoltageArray& getAdaptiveApdVoltageArray() { return m_adaptiveApdVoltageArray; }
    float getMinApdVoltageOffset() const { return m_minApdVoltageOffset; }
    float getMaxApdVoltageOffset() const { return m_maxApdVoltageOffset; }
    float getNoiseMeasurementThreshold() const { return m_noiseMeasurementThreshold; }
    uint16_t getReferenceNoise() const { return m_referenceNoise; }
    const ActualNoiseArray& getActualNoiseArray() const { return m_actualNoiseArray; }
    ActualNoiseArray& getActualNoiseArray() { return m_actualNoiseArray; }

public: // setter
    void setSerialNumberOfScanner(const SerialNumber& serialNumberOfScanner)
    {
        m_serialNumberOfScanner = serialNumberOfScanner;
    }
    void setScannerType(ScannerType scannerType) { m_scannerType = scannerType; }
    void setDeviceId(uint8_t deviceId) { m_deviceId = deviceId; }
    void setFpgaVersion(const Version448& fpgaVersion) { m_fpgaVersion = fpgaVersion; }
    void setDspVersion(const Version448& dspVersion) { m_dspVersion = dspVersion; }
    void setHostVersion(const Version448& hostVersion) { m_hostVersion = hostVersion; }
    void setFpgaOperationRegister(uint16_t fpgaOperationRegister) { m_fpgaOperationRegister = fpgaOperationRegister; }
    void setFpgaStatusRegister(uint16_t fpgaStatusRegister) { m_fpgaStatusRegister = fpgaStatusRegister; }
    void setSensorTemperatureApd0(float sensorTemperatureApd0) { m_sensorTemperatureApd0 = sensorTemperatureApd0; }
    void setSensorTemperatureApd1(float sensorTemperatureApd1) { m_sensorTemperatureApd1 = sensorTemperatureApd1; }
    void setScanFrequency(float frequency) { m_scanFrequency = frequency; }
    void setApdTableBiasVoltage(float apdTableBiasVoltage) { m_apdTableBiasVoltage = apdTableBiasVoltage; }
    void setAdaptiveApdVoltageArray(const AdaptiveApdVoltageArray& adaptiveApdVoltageArray)
    {
        m_adaptiveApdVoltageArray = adaptiveApdVoltageArray;
    }
    void setMinApdVoltageOffset(float minApdVoltageOffset) { m_minApdVoltageOffset = minApdVoltageOffset; }
    void setMaxApdVoltageOffset(float maxApdVoltageOffset) { m_maxApdVoltageOffset = maxApdVoltageOffset; }
    void setNoiseMeasurementThreshold(float noiseMeasurementThreshold)
    {
        m_noiseMeasurementThreshold = noiseMeasurementThreshold;
    }
    void setReferenceNoise(uint16_t referenceNoise) { m_referenceNoise = referenceNoise; }
    void setActualNoiseArray(const ActualNoiseArray& actualNoiseArray) { m_actualNoiseArray = actualNoiseArray; }

protected:
    SerialNumber m_serialNumberOfScanner{}; //!< Serial number of the scanner.
    ScannerType m_scannerType{}; //!< Type of the scanner.
    uint8_t m_deviceId{0}; //!< Device id of the scanner.

    Version448 m_fpgaVersion{}; //!< Version of the FPGA.
    Version448 m_dspVersion{}; //!< Version of the DSP.
    Version448 m_hostVersion{}; //!< Version of the host.

    uint16_t m_fpgaOperationRegister{}; //!< State of the FPGA operation register.
    uint16_t m_fpgaStatusRegister{}; //!< State of the FPGA modus register.
    float m_sensorTemperatureApd0{0.0F}; //!< Sensor APD temperature 0 [K].
    float m_sensorTemperatureApd1{0.0F}; //!< Sensor APD temperature 1 [K].
    float m_scanFrequency{0.0F}; //!< Scan frequency [Hz].
    float m_apdTableBiasVoltage{0.0F}; // APD bias voltage [V].
    AdaptiveApdVoltageArray m_adaptiveApdVoltageArray; // APD voltage [V].
    float m_minApdVoltageOffset{0.0F}; //!< Minimal APD voltage offset.
    float m_maxApdVoltageOffset{0.0F}; //!< Maximal APD voltage offset.
    float m_noiseMeasurementThreshold{0.0F}; //!< Noise measurement threshold [V].
    uint16_t m_referenceNoise{0}; //!< Reference noise.
    ActualNoiseArray m_actualNoiseArray;
}; // DeviceStatus

//==============================================================================

bool operator==(const DeviceStatus& lhs, const DeviceStatus& rhs);
inline bool operator!=(const DeviceStatus& lhs, const DeviceStatus& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
