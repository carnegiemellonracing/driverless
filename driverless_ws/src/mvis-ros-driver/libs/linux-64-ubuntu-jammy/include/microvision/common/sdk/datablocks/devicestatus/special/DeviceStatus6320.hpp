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
//! \date Jun 24, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/ErrorIn6320.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Device Status of MVIS SyncBox
//==============================================================================
class DeviceStatus6320 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using MacAddress  = std::array<uint8_t, 6>;
    using ErrorVector = std::vector<ErrorIn6320>;

    //========================================
    //! \brief Source providing the time for the SyncBox.
    //----------------------------------------
    enum class TimeSyncSource : uint8_t
    {
        Oscillator       = 0x00U, //!<  The time source is from a Oscillator.
        Ptp              = 0x01U, //!< The time source is from Ptp.
        Gptp             = 0x02U, //!< The time source is from Gptp.
        Gps              = 0x03U, //!< The time source is from a Gps.
        GpsAndOscillator = 0x04U, //!< The time source is from Gps and Oscillator.
        Ntp              = 0x05U, //!< The time source is from Ntp.
        Can              = 0x06U, //!< The time source is from a Can.
        Error            = 0xFFU //!< The time source has a Error.
    };

    //========================================
    //! \brief State of the synchronization.
    //----------------------------------------
    enum class TimeSyncState : uint8_t
    {
        Unknown          = 0x00U, //!< The state is unknown.
        Initialized      = 0x01U, //!< The state is initialized.
        Active           = 0x02U, //!< The state is active.
        WaitForFirstSync = 0x03U, //!< The state is waiting for the first sync.
        InSync           = 0x04U, //!< The state is in sync.
        OutOfSync        = 0x05U, //!< The state is out of sync.
        SyncTimeout      = 0x06U, //!< The state has a sync timeout.
        Error            = 0xFFU //!< The state hat an error.
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.devicestatus6320"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    DeviceStatus6320() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~DeviceStatus6320() = default;

public: // getter
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

    //========================================
    //! \brief Get the message type.
    //!
    //! \return The message type.
    //! \note Currently, only message type 0x0001 (heartbeat message) is supported.
    //----------------------------------------
    uint16_t getMessageType() const { return m_messageType; }

    //========================================
    //! \brief Get the MAC address of the SyncBox.
    //!
    //! \return The MAC address of the SyncBox.
    //----------------------------------------
    const MacAddress& getMacAddress() const { return m_macAddress; }

    //========================================
    //! \brief Get the heartbeat count.
    //!
    //! \return The heartbeat count which is incremented for each heartbeat message.
    //----------------------------------------
    uint16_t getHeartbeatCount() const { return m_heartbeatCount; }

    //========================================
    //! \brief Get the current source providing the time.
    //!
    //! \return The source currently providing the time.
    //----------------------------------------
    TimeSyncSource getTimeSyncSource() const { return m_timeSyncSource; }

    //========================================
    //! \brief Get the state of the synchronization.
    //!
    //! \return The state of the synchronization.
    //----------------------------------------
    TimeSyncState getTimeSyncState() const { return m_timeSyncState; }

    //========================================
    //! \brief Get the time when the last synchronization happened.
    //!
    //! \return The time when the last synchronization happened.
    //----------------------------------------
    const NtpTime& getLastSyncTimestamp() const { return m_lastSyncTimestamp; }

    //========================================
    //! \brief Get the delta in time that was corrected by the last synchronization.
    //!
    //! \return The delta in time that was corrected by the last synchronization.
    //----------------------------------------
    const NtpTime& getLastSyncDelta() const { return m_lastSyncDelta; }

    //========================================
    //! \brief Get the number of cycles spent in the device's main loop.
    //!
    //! \return The number of cycles spent in the device's main loop.
    //----------------------------------------
    uint32_t getMainLoopCycleCount() const { return m_mainLoopCycleCount; }

    //========================================
    //! \brief Get the errors that have been detected since last heartbeat message.
    //!
    //! \return The errors that have been detected since last heartbeat message.
    //----------------------------------------
    const ErrorVector& getErrors() const { return m_errors; }

    //========================================
    //! \brief Get the errors that have been detected since last heartbeat message.
    //!
    //! \return The errors that have been detected since last heartbeat message.
    //----------------------------------------
    ErrorVector& getErrors() { return m_errors; }

public: // setter
    //========================================
    //! \brief Set the message type
    //!
    //! \param[in] messageType  The new message type.
    //! \note Currently, only message type 0x0001 (heartbeat message) is supported.
    //----------------------------------------
    void setMessageType(const uint16_t messageType) { m_messageType = messageType; }

    //========================================
    //! \brief Set the MAC address of the SyncBox.
    //!
    //! \param[in] macAddress  The new MAC address of the SyncBox.
    //----------------------------------------
    void setMacAddress(const MacAddress& macAddress) { m_macAddress = macAddress; }

    //========================================
    //! \brief Set the heartbeat count.
    //!
    //! \param[in] heartbeatCount  The new heartbeat count.
    //----------------------------------------
    void setHeartbeatCount(const uint16_t heartbeatCount) { m_heartbeatCount = heartbeatCount; }

    //========================================
    //! \brief Set the current source providing the time.
    //!
    //! \param[in] timeSyncSource  The new source currently providing the time.
    //----------------------------------------
    void setTimeSyncSource(const TimeSyncSource timeSyncSource) { m_timeSyncSource = timeSyncSource; }

    //========================================
    //! \brief Set the state of the synchronization.
    //!
    //! \param[in] timeSyncState  The new state of the synchronization.
    //----------------------------------------
    void setTimeSyncState(const TimeSyncState timeSyncState) { m_timeSyncState = timeSyncState; }

    //========================================
    //! \brief Set the time when the last synchronization happened.
    //!
    //! \param[in] lastSyncTimestamp  The new time when the last synchronization happened.
    //----------------------------------------
    void setLastSyncTimestamp(const NtpTime& lastSyncTimestamp) { m_lastSyncTimestamp = lastSyncTimestamp; }

    //========================================
    //! \brief Set the delta in time that was corrected by the last synchronization.
    //!
    //! \param[in] lastSyncDelta  The new delta in time that was corrected by the last synchronization.
    //----------------------------------------
    void setLastSyncDelta(const NtpTime& lastSyncDelta) { m_lastSyncDelta = lastSyncDelta; }

    //========================================
    //! \brief Set the number of cycles spent in the device's main loop.
    //!
    //! \param[in] mainLoopCycleCount  The new number of cycles spent in the device's main loop.
    //----------------------------------------
    void setMainLoopCycleCount(const uint32_t mainLoopCycleCount) { m_mainLoopCycleCount = mainLoopCycleCount; }

    //========================================
    //! \brief Set the errors that have been detected since last heartbeat message.
    //!
    //! \param[in] errors  The new errors that have been detected since last heartbeat message.
    //----------------------------------------
    void setErrors(const ErrorVector& errors) { m_errors = errors; }

protected:
    uint16_t m_messageType{0};
    MacAddress m_macAddress{{0, 0, 0, 0, 0, 0}};
    uint16_t m_heartbeatCount{0};
    TimeSyncSource m_timeSyncSource{TimeSyncSource::Error};
    TimeSyncState m_timeSyncState{TimeSyncState::Error};
    NtpTime m_lastSyncTimestamp{0};
    NtpTime m_lastSyncDelta{0};
    uint32_t m_mainLoopCycleCount{0};
    ErrorVector m_errors;
}; // DeviceStatus6320

//==============================================================================

//==============================================================================
//! \brief Test DeviceStatus6320 objects for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise
//------------------------------------------------------------------------------
bool operator==(const DeviceStatus6320& lhs, const DeviceStatus6320& rhs);

//==============================================================================
//! \brief Test DeviceStatus6320 objects for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const DeviceStatus6320& lhs, const DeviceStatus6320& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
