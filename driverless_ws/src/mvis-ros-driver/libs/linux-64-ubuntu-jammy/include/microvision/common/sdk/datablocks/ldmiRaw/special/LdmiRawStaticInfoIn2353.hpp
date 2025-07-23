//==============================================================================
//! \file
//!
//! \brief Ldmi static info helper for reassembling ldmi raw frame 2353 package.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 28th, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrameHeaderIn2353.hpp>

#include <microvision/common/logging/logging.hpp>

#include <array>
#include <sstream>
#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Ldmi static info helper for reassembling ldmi raw frame package.
//------------------------------------------------------------------------------
class LdmiRawStaticInfoIn2353 final
{
public:
    //========================================
    //! \brief List type of frame 'Number of rows' configurations by power mode.
    //----------------------------------------
    using FrameConfigByModeType = std::array<uint16_t, LdmiRawFrameHeaderIn2353::countOfPowerModes>;

    //========================================
    //! \brief Size of whole message.
    //----------------------------------------
    static constexpr std::size_t payloadSize{10170U};

    //========================================
    //! \brief Offset where the sensor type is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadSensorTypeOffset{190U};

    //========================================
    //! \brief Offset where the reference id is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadIdOffset{192U};

    //========================================
    //! \brief Offset where the 'Number of row messages' (not number of rows)
    //! by normal power mode is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadNbOfRowsByNormalPowerOffset{1947U + 2U};

    //========================================
    //! \brief Offset where the 'Number of row messages' (not number of rows)
    //! by normal low mode is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadNbOfRowsByLowPowerOffset{1947U + 2741U + 2U};

    //========================================
    //! \brief Offset where the 'Number of row messages' (not number of rows)
    //! by high power mode is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadNbOfRowsByHighPowerOffset{1947U + 2741U + 2741U + 2U};

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::LdmiRawStaticInfoIn2353";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    LdmiRawStaticInfoIn2353();

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another instance of frame static info.
    //----------------------------------------
    LdmiRawStaticInfoIn2353(LdmiRawStaticInfoIn2353&& other);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another instance of frame static info.
    //----------------------------------------
    LdmiRawStaticInfoIn2353(const LdmiRawStaticInfoIn2353& other);

    //========================================
    //! \brief Construct instance by message payload of LdmiA300 and deserialize
    //! information for reassembling.
    //! \param[in] messagePayload  LdmiA300 message payload.
    //----------------------------------------
    explicit LdmiRawStaticInfoIn2353(const SharedBuffer& messagePayload);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LdmiRawStaticInfoIn2353() = default;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Another instance of frame static info.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawStaticInfoIn2353& operator=(LdmiRawStaticInfoIn2353&& other);

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Another instance of frame static info.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawStaticInfoIn2353& operator=(const LdmiRawStaticInfoIn2353& other);

public:
    //========================================
    //! \brief Compares two ldmi raw static infos for equality.
    //! \param[in] lhs  Ldmi raw static info in 2353.
    //! \param[in] rhs  Ldmi raw static info in 2353.
    //! \return Either \c true if both static infos are equal or otherwise \c
    //! false.
    //----------------------------------------
    friend bool operator==(const LdmiRawStaticInfoIn2353& lhs, const LdmiRawStaticInfoIn2353& rhs);

    //========================================
    //! \brief Compares two ldmi raw static infos for inequality.
    //! \param[in] lhs  Ldmi raw static info in 2353.
    //! \param[in] rhs  Ldmi raw static info in 2353.
    //! \return Either \c true if both static infos are unequal or otherwise \c
    //! false.
    //----------------------------------------
    friend bool operator!=(const LdmiRawStaticInfoIn2353& lhs, const LdmiRawStaticInfoIn2353& rhs);

public: // getter
    //========================================
    //! \brief Get LdmiA300 message payload.
    //!
    //! Serialized form of static info which contains frame id and more.
    //!
    //! \return LdmiA300 message payload.
    //----------------------------------------
    const SharedBuffer& getMessagePayload() const;

    //========================================
    //! \brief Get sensor type.
    //! \return Sensor type.
    //----------------------------------------
    uint16_t getSensorType() const;

    //========================================
    //! \brief Get reference id of static info.
    //! \return Reference if of static info.
    //----------------------------------------
    uint64_t getId() const;

    //========================================
    //! \brief Get array of 'Number of rows' by frame power mode.
    //! \return Array of 'Number of rows' by frame power mode.
    //----------------------------------------
    FrameConfigByModeType getNumberOfRows() const;

public: // setter
    //========================================
    //! \brief Set LdmiA300 message payload and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA300 message payload.
    //----------------------------------------
    void setMessagePayload(const SharedBuffer& messagePayload);

private:
    //========================================
    //! \brief Message payload.
    //----------------------------------------
    SharedBuffer m_messagePayload;

    //========================================
    //! \brief Sensor type stored in static info.
    //----------------------------------------
    uint16_t m_sensorType;

    //========================================
    //! \brief Reference id of static info.
    //----------------------------------------
    uint64_t m_id;

    //========================================
    //! \brief Number of rows by frame power mode.
    //----------------------------------------
    FrameConfigByModeType m_nbOfRows;
}; // LdmiRawStaticInfoIn2353

//==============================================================================

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
