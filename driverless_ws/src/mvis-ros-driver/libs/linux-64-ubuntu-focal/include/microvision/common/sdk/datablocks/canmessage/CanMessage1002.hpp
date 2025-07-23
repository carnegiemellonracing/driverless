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
//! \date Jan 12, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//! \brief CAN Message over Ethernet
//!
//! Messages are tunneled over Ethernet
//!
//! General data type: \ref microvision::common::sdk::CanMessage
//------------------------------------------------------------------------------
class CanMessage1002 : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const int maxVersion        = 15;
    static const uint8_t maxMsgSize    = 8U;
    static const uint32_t maxStdId     = 0x7FFU;
    static const uint32_t maxExtId     = 0x1FFFFFFFU; //! Maximum extended CAN identifier
    static const uint32_t stdTsBitMask = 0x00008000U;
    static const uint32_t extTsBitMask = 0x80000000U;

public:
    using DataArray = std::array<uint8_t, maxMsgSize>;

public:
    //========================================
    //! \brief CAN message types
    //! \note These message types have been extracted from PCAN header files.
    //----------------------------------------

    enum class MsgType : uint8_t
    {
        Standard = 0x00U, //! Standard data frame (11-bit ID)
        RTR      = 0x01U, //! Remote request frame
        Extended = 0x02U, //! Extended data frame (CAN 2.0B, 29-bit ID)
        ErrFrame = 0x40U, //! Error frame
        Status   = 0x80U //! Status information
    };

    //========================================
    //!brief Enumeration of byte numbers 0 to 7
    //----------------------------------------
    enum class ByteNumber : uint8_t
    {
        Byte0 = 0,
        Byte1 = 1,
        Byte2 = 2,
        Byte3 = 3,
        Byte4 = 4,
        Byte5 = 5,
        Byte6 = 6,
        Byte7 = 7
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.canmessage1002"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CanMessage1002();
    ~CanMessage1002() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    bool hasExtendedCanId() const
    {
        return ((static_cast<uint8_t>(this->m_msgType) & static_cast<uint8_t>(MsgType::Extended))
                == static_cast<uint8_t>(MsgType::Extended));
    }
    bool hasValidTimestamp() const { return !this->m_timestamp.toPtime().is_not_a_date_time(); }
    bool hasTimeStamp() const
    {
        const uint32_t mask = hasExtendedCanId() ? extTsBitMask : stdTsBitMask;
        return ((m_canId & mask) == mask);
    }

public: // getter
    uint8_t getVersion() const { return m_version; }
    uint8_t getLength() const { return m_length; }
    uint8_t getData(const ByteNumber byte) const { return m_data[static_cast<uint8_t>(byte)]; }
    MsgType getMsgType() const { return m_msgType; }
    uint32_t getCanId() const { return m_canId; }
    uint32_t getUsSinceStartup() const { return m_usSinceStartup; }
    NtpTime getTimestamp() const { return m_timestamp; }
    uint8_t getDeviceId() const { return m_deviceId; }

public: // setter
    void setVersion(const uint8_t newVersion) { m_version = newVersion; }
    void setLength(const uint8_t newLength) { m_length = newLength; }
    void setData(const ByteNumber byte, const uint8_t newData) { m_data[static_cast<uint8_t>(byte)] = newData; }
    void setMsgType(const MsgType newMsgType) { m_msgType = newMsgType; }
    void setCanId(const uint32_t newCanId) { m_canId = newCanId; }
    void setUsSinceStartup(const uint32_t newUsSinceStartup) { m_usSinceStartup = newUsSinceStartup; }
    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }

protected:
    uint8_t m_version{0}; //!<  The Version number of the message. Serialized as 4 bits.
    uint8_t m_length{0}; //!< The length of the message data. Serialized as  4 bits

    DataArray m_data{{}}; //!< The data of this message.
    MsgType m_msgType{MsgType::Standard}; //!< The type of the message.
    uint32_t m_canId{0}; //!< The CAN-ID is an 11-bit-identifier. serialized as 2 or 4 bytes

    uint32_t m_usSinceStartup{0}; //!< Microseconds since device startup.
    NtpTime m_timestamp{0}; //!< Time when message has been received by remote receiver.

    uint8_t m_deviceId{0}; //! id of device
}; // CanMessage1002

//==============================================================================

bool operator==(const CanMessage1002& lhs, const CanMessage1002& rhs);
bool operator!=(const CanMessage1002& lhs, const CanMessage1002& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
