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
//! \date Dec 3rd, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SharedBuffer.hpp>

#include <microvision/common/sdk/Ptp96Time.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Ldmi frame header helper for reassembling ldmi raw frame package.
//------------------------------------------------------------------------------
class LdmiRawFrameHeaderIn2352 final
{
public:
    //========================================
    //! \brief Size of whole message.
    //----------------------------------------
    static constexpr std::size_t payloadSize{128U};

    //========================================
    //! \brief Offset where the frame id is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadFrameIdOffset{4U};

    //========================================
    //! \brief Number of supported power modes.
    //----------------------------------------
    static constexpr std::size_t countOfPowerModes{3U};

    //========================================
    //! \brief Power modes.
    //----------------------------------------
    enum class PowerMode : uint8_t
    {
        Normal    = 0x00U,
        LowPower  = 0x01U,
        HighPower = 0x02U
    };

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::LdmiRawFrameHeaderIn2352";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    LdmiRawFrameHeaderIn2352();

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another instance of frame header.
    //----------------------------------------
    LdmiRawFrameHeaderIn2352(LdmiRawFrameHeaderIn2352&& other);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another instance of frame header.
    //----------------------------------------
    LdmiRawFrameHeaderIn2352(const LdmiRawFrameHeaderIn2352& other);

    //========================================
    //! \brief Construct instance by message payload of LdmiA2F0 and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA2F0 message payload.
    //----------------------------------------
    explicit LdmiRawFrameHeaderIn2352(const SharedBuffer& messagePayload);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LdmiRawFrameHeaderIn2352() = default;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Another instance of frame header.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawFrameHeaderIn2352& operator=(LdmiRawFrameHeaderIn2352&& other);

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Another instance of frame header.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawFrameHeaderIn2352& operator=(const LdmiRawFrameHeaderIn2352& other);

public:
    //========================================
    //! \brief Compares two ldmi raw frame headers for equality.
    //! \param[in] lhs  Ldmi raw frame header in 2352.
    //! \param[in] rhs  Ldmi raw frame header in 2352.
    //! \return Either \c true if both frame headers are equal or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const LdmiRawFrameHeaderIn2352& lhs, const LdmiRawFrameHeaderIn2352& rhs);

    //========================================
    //! \brief Compares two ldmi raw frame headers for inequality.
    //! \param[in] lhs  Ldmi raw frame header in 2352.
    //! \param[in] rhs  Ldmi raw frame header in 2352.
    //! \return Either \c true if both frame headers are unequal or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const LdmiRawFrameHeaderIn2352& lhs, const LdmiRawFrameHeaderIn2352& rhs);

public: // getter
    //========================================
    //! \brief Get LdmiA2F0 message payload.
    //!
    //! Serialized form of frame header which contains frame id and more.
    //!
    //! \return LdmiA2F0 message payload.
    //----------------------------------------
    const SharedBuffer& getMessagePayload() const;

    //========================================
    //! \brief Get power mode.
    //! \return Power mode.
    //----------------------------------------
    PowerMode getMode() const;

    //========================================
    //! \brief Get frame id.
    //! \return Frame id.
    //----------------------------------------
    uint32_t getFrameId() const;

    //========================================
    //! \brief Get reference id of static configuration.
    //! \return Reference id.
    //----------------------------------------
    uint64_t getConfigId() const;

    //========================================
    //! \brief Get timestamp of frame start.
    //! \return Frame timestamp.
    //----------------------------------------
    Ptp96Time getTimestamp() const;

public: // setter
    //========================================
    //! \brief Set LdmiA2F0 message payload and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA2F0 message payload.
    //----------------------------------------
    void setMessagePayload(const SharedBuffer& messagePayload);

private:
    //========================================
    //! \brief Message payload.
    //----------------------------------------
    SharedBuffer m_messagePayload;

    //========================================
    //! \brief Frame timestamp.
    //----------------------------------------
    Ptp96Time m_timestamp;

    //========================================
    //! \brief Refrence id.
    //----------------------------------------
    uint64_t m_configId;

    //========================================
    //! \brief Frame id.
    //----------------------------------------
    uint32_t m_frameId;

    //========================================
    //! \brief Power mode.
    //----------------------------------------
    PowerMode m_mode;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
