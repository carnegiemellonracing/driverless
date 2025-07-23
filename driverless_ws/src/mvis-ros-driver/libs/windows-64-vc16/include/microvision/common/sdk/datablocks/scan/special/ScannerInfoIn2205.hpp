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
//! \date Apr 26, 2012
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ResolutionInfoIn2205.hpp>
#include <microvision/common/sdk/RotationOrder.hpp>
#include <microvision/common/sdk/ScannerType.hpp>
#include <microvision/common/sdk/TransformationMatrix3d.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ScannerInfoIn2205 final
{
public:
    enum class ScannerStatusFlags : uint32_t
    {
        MotorSyncActive       = 0x00000010U,
        MotorSyncLocked       = 0x00000020U,
        SyncMasterMode        = 0x00000040U,
        SyncSignal            = 0x00001000U,
        SyncPpsSignal         = 0x00002000U,
        ValidFreqSyncDetected = 0x00003000U,
        UpsideDownLux         = 0x00004000U,

        NotifyChange     = SyncPpsSignal | MotorSyncLocked | SyncSignal,
        PpsSyncActive    = SyncPpsSignal | MotorSyncLocked,
        MaskSyncDetected = SyncPpsSignal | SyncSignal | ValidFreqSyncDetected
    };
    //TODO: MaskSyncDetected is the same as ValidFreqSyncDetected !?!

public:
    // The scanner info flags are an extension to the scan flags.
    static constexpr uint32_t rotationFlagsShift = 14; //!< Rotation flags start at bit 14 in the scanner info flags.
    static constexpr uint32_t rotationFlagsMask  = 0x07; //!< Size of rotation flags: 3 bit.

public:
    ScannerInfoIn2205();
    ScannerInfoIn2205(const ScannerInfoIn2205& src);
    virtual ~ScannerInfoIn2205();

    ScannerInfoIn2205& operator=(const ScannerInfoIn2205& other);

public:
    static std::streamsize getSerializedSize_static();

public:
    std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public:
    uint8_t getDeviceId() const { return m_deviceId; }
    ScannerType getScannerType() const { return m_scannerType; }
    uint16_t getScanNumber() const { return m_scanNumber; }

    uint32_t getScannerStatus() const { return m_scannerStatus; }

    bool isScannerStatusFlagSet(const ScannerStatusFlags flag) const;
    bool isSyncSignal() const;
    bool isSyncPpsSignal() const;
    bool isMotorSyncActive() const;
    bool isMotorInPhase() const;
    bool isUpsideDownLux() const;

    float getStartAngle() const { return m_startAngle; }
    float getEndAngle() const { return m_endAngle; }

    NtpTime getScanStartTime() const { return m_scanStartTime; }
    NtpTime getScanEndTime() const { return m_scanEndTime; }

    NtpTime getScanStartTimeFromDevice() const { return m_scanStartTimeFromDevice; }
    NtpTime getScanEndTimeFromDevice() const { return m_scanEndTimeFromDevice; }

    float getFrequency() const { return m_scanFrequency; }
    float getBeamTilt() const { return m_beamTilt; }

    uint32_t getFlags() const { return m_scanFlags; }

    bool areFlagsSet(const uint32_t flags) const;

    bool isGroundLabeled() const;
    bool isDirtLabeled() const;
    bool isRainLabeled() const;
    bool isCoverageLabeled() const;
    bool isBackgroundLabeled() const;
    bool isReflectorLabeled() const;
    bool isUpsideDown() const;
    bool isRearMirrorSide() const;

    float getYawAngle() const { return m_yawAngle; }
    float getPitchAngle() const { return m_pitchAngle; }
    float getRollAngle() const { return m_rollAngle; }

    float getOffsetX() const { return m_offsetX; }
    float getOffsetY() const { return m_offsetY; }
    float getOffsetZ() const { return m_offsetZ; }

    RotationOrder getRotationOrder() const
    {
        return static_cast<RotationOrder>((m_scanFlags >> rotationFlagsShift) & rotationFlagsMask);
    }

    //========================================
    //! \brief Create a matrix for converting points from this scanner's coordinate system into vehicle coordinate
    //! system.
    //!
    //! \param[in] initialTransformationMatrix  Rotation matrix for initial transformation (e.g. for adjusting the zero
    //!                                         degree mark on the sensor like for some third party lidar sensors).
    //! \return Matrix for transforming coordinates into vehicle coordinate system.
    //----------------------------------------
    TransformationMatrix3d<float>
    getFromSensorToVehicleTransformation(const Matrix3x3<float>& initialTransformationMatrix
                                         = Matrix3x3<float>()) const;

    const std::vector<ResolutionInfoIn2205>& getResolutionInfo() const { return m_ri; }
    std::vector<ResolutionInfoIn2205>& getResolutionInfo() { return m_ri; }

public:
    void setDeviceId(const uint8_t newDeviceID) { m_deviceId = newDeviceID; }
    void setScannerType(const ScannerType newScannerType) { m_scannerType = newScannerType; }
    void setScanNumber(const uint16_t newScanNumber) { m_scanNumber = newScanNumber; }

    void setScannerStatus(const uint32_t newScannerStatus) { m_scannerStatus = newScannerStatus; }

    void setStartAngle(const float newStartAngle) { m_startAngle = newStartAngle; }
    void setEndAngle(const float newEndAngle) { m_endAngle = newEndAngle; }

    void setScanStartTime(const NtpTime newScanStartTime) { m_scanStartTime = newScanStartTime; }
    void setScanEndTime(const NtpTime newScanEndTime) { m_scanEndTime = newScanEndTime; }

    void setScanStartTimeFromDevice(const NtpTime newScanStartTimeFromDevice)
    {
        m_scanStartTimeFromDevice = newScanStartTimeFromDevice;
    }
    void setScanEndTimeFromDevice(const NtpTime newScanEndTimeFromDevice)
    {
        m_scanEndTimeFromDevice = newScanEndTimeFromDevice;
    }

    void setFrequency(const float newFrequency) { m_scanFrequency = newFrequency; }
    void setBeamTilt(const float newBeamTilt) { m_beamTilt = newBeamTilt; }

    void setFlags(const uint32_t newScanFlags) { m_scanFlags = newScanFlags; }

    void setYawAngle(const float newYawAngle) { m_yawAngle = newYawAngle; }
    void setPitchAngle(const float newPitchAngle) { m_pitchAngle = newPitchAngle; }
    void setRollAngle(const float newRollAngle) { m_rollAngle = newRollAngle; }

    void setOffsetX(const float newOffsetX) { m_offsetX = newOffsetX; }
    void setOffsetY(const float newOffsetY) { m_offsetY = newOffsetY; }
    void setOffsetZ(const float newOffsetZ) { m_offsetZ = newOffsetZ; }

    void setRotationOrder(const RotationOrder rotationOrder)
    {
        // rotation order is stored in flag bits
        const auto mask = ~(rotationFlagsMask << rotationFlagsShift);
        m_scanFlags &= mask;
        const auto flag = static_cast<uint32_t>(rotationOrder) << rotationFlagsShift;
        m_scanFlags |= flag;
    }

    void setResolutionInfo(const std::vector<ResolutionInfoIn2205> newResolutionInfo) { m_ri = newResolutionInfo; }

public:
    static const unsigned int nbOfResolutionInfo = 8;

public:
    bool operator==(const ScannerInfoIn2205& other) const;
    bool operator!=(const ScannerInfoIn2205& other) const { return !((*this) == other); }

protected:
    uint8_t m_deviceId; //!< Device ID of this scanner.
    ScannerType m_scannerType; //!< The scanner type which is used.
    uint16_t m_scanNumber; //!< The scan number coming from the scanner device.

    uint32_t m_scannerStatus; //!< The scanner status from the ScannerStatusFlags.

    float
        m_startAngle; //!< Field of view of this scanner given in its local coordinate system. [rad] - between [-pi;pi]
    float m_endAngle; //!< Field of view of this scanner given in its local coordinate system. [rad] - between [-pi;pi]

    //! NtpTime when the first measurement of this scanner was done. Based on computer time.
    NtpTime m_scanStartTime;
    //! NtpTime when the last measurement of this scanner was done. Based on computer time.
    NtpTime m_scanEndTime;
    //! NtpTime when the first measurement of this scanner was done. Based on the sensor time.
    NtpTime m_scanStartTimeFromDevice;
    //! NtpTime when the last measurement of this scanner was done. Based on the sensor time.
    NtpTime m_scanEndTimeFromDevice;

    float m_scanFrequency; //!< Scan frequency of this scanner in Hz. [Hz]

    //! Angle of the beam Tilt. [rad] - between [-pi;pi]
    //! \note The scanner measurement is pitched relatively to sensor x-y plane. This value is valid for measuring in
    //!       x-direction resp. 0° in the scanner coordinate system.  In radians normalized to \( [-\pi,+\pi] \).
    //!       Beam is pitched downwards if values are positive and vice versa.
    float m_beamTilt;

    uint32_t m_scanFlags; //!< The scanner info flags are an extension to the scan flags.
    float m_yawAngle; //!<  Mounting angles relative to vehicle coordinate system. [rad] - between [-pi;pi]
    float m_pitchAngle; //!< Mounting angles relative to vehicle coordinate system. [rad] - between [-pi;pi]
    float m_rollAngle; //!<  Mounting angles relative to vehicle coordinate system. [rad] - between [-pi;pi]
    float m_offsetX; //!< Mounting position relative to vehicle coordinate system in meters. [m]
    float m_offsetY; //!< Mounting position relative to vehicle coordinate system in meters. [m]
    float m_offsetZ; //!< Mounting position relative to vehicle coordinate system in meters. [m]
    std::vector<ResolutionInfoIn2205> m_ri; //!< Scan resolution for different sectors of the scanner field of view.
}; // ScannerInfo

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
