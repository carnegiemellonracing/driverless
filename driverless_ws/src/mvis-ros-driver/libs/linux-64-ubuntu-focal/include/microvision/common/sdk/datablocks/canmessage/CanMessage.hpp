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
//! \date 12.October 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Math.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief CAN Message
//!
//! Special data type: \ref microvision::common::sdk::CanMessage1002
//------------------------------------------------------------------------------
class CanMessage final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const int maxVersion        = 15;
    static const uint8_t maxMsgSize    = 8;
    static const uint32_t maxStdId     = 0x7FFU;
    static const uint32_t maxExtId     = 0x1FFFFFFFU; //!< Maximum extended CAN identifier.
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
    //! \brief Enumeration of byte numbers 0 to 7.
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
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.canmessage"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    CanMessage() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~CanMessage() override = default;

public: // DataContainerBase implementation
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
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
    //========================================
    //! \brief Get the version of the can message.
    //! \return The version.
    //----------------------------------------
    uint8_t getVersion() const { return m_version; }

    //========================================
    //! \brief Get the length of the can message.
    //! \return The length.
    //----------------------------------------
    uint8_t getLength() const { return m_length; }

    //========================================
    //! \brief Get the data of the can message.
    //! \param[in] byte  The byte position of the requested data.
    //! \return The data at the specific position.
    //----------------------------------------
    uint8_t getData(const ByteNumber byte) const { return m_data.at(static_cast<uint8_t>(byte)); }

    //========================================
    //! \brief Get the message type of the can message.
    //! \return The message type.
    //----------------------------------------
    MsgType getMsgType() const { return m_msgType; }

    //========================================
    //! \brief Get the can id of the can message.
    //! \return The can id.
    //----------------------------------------
    uint32_t getCanId() const { return m_canId; }

    //========================================
    //! \brief Get the microseconds since startup in [us].
    //! \return The microseconds since startup.
    //----------------------------------------
    uint32_t getUsSinceStartup() const { return m_usSinceStartup; }

    //========================================
    //! \brief Get the timestamp of the can message.
    //! \return The timestamp.
    //----------------------------------------
    NtpTime getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the device id.
    //! \return The device id.
    //----------------------------------------
    uint8_t getDeviceId() const { return m_deviceId; }

public: // setter
    //========================================
    //! \brief Set the version of the can message.
    //! \param[in] newVersion  The new version of the can message.
    //----------------------------------------
    void setVersion(const uint8_t newVersion) { m_version = newVersion; }

    //========================================
    //! \brief Set the length of the can message.
    //! \param[in] newLength  The new length of the can message.
    //----------------------------------------
    void setLength(const uint8_t newLength) { m_length = std::min(newLength, static_cast<uint8_t>(maxMsgSize)); }

    //========================================
    //! \brief Set the data of the can message.
    //! \param[in] byte  The byte position of the new data.
    //! \param[in] newData  The new data of the can message.
    //----------------------------------------
    void setData(const ByteNumber byte, const uint8_t newData) { m_data.at(static_cast<uint8_t>(byte)) = newData; }

    //========================================
    //! \brief Set the message type of the can message.
    //! \param[in] newMsgType  The new message type of the can message.
    //----------------------------------------
    void setMsgType(const MsgType newMsgType) { m_msgType = newMsgType; }

    //========================================
    //! \brief Set the can id.
    //! \param[in] newCanId  The new can id of the can message.
    //----------------------------------------
    void setCanId(const uint32_t newCanId) { m_canId = newCanId; }

    //========================================
    //! \brief Set the microseconds since startup.
    //! \param[in] newUsSinceStartup  The new microseconds since startup in [us].
    //----------------------------------------
    void setUsSinceStartup(const uint32_t newUsSinceStartup) { m_usSinceStartup = newUsSinceStartup; }

    //========================================
    //! \brief Set the timestamp of the can message.
    //! \param[in] newTimestamp  The new timestamp of the can message.
    //----------------------------------------
    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }

    //========================================
    //! \brief Set the device id of the can message.
    //! \param[in] newDeviceId  The new device id of the can message.
    //----------------------------------------
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }

protected: //members
    uint8_t m_version{0}; // 4 bits
    uint8_t m_length{0}; // 4 bits

    DataArray m_data{{0}};
    MsgType m_msgType{MsgType::Standard};
    uint32_t m_canId{0}; // serialized as 2 or 4 bytes

    uint32_t m_usSinceStartup{0}; //!< Microseconds since device startup.
    NtpTime m_timestamp{0};

    uint8_t m_deviceId{0}; //!< id of device

}; //CanMessage
//==============================================================================

bool operator==(const CanMessage& lhs, const CanMessage& rhs);
bool operator!=(const CanMessage& lhs, const CanMessage& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
