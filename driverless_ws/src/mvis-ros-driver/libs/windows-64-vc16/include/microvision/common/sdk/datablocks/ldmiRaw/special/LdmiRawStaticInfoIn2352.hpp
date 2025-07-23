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

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrameHeaderIn2352.hpp>

#include <microvision/common/logging/logging.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Ldmi static info helper for reassembling ldmi raw frame package.
//------------------------------------------------------------------------------
class LdmiRawStaticInfoIn2352 final
{
public:
    //========================================
    //! \brief List type of frame 'Number of rows' configurations by power mode.
    //----------------------------------------
    using FrameConfigByModeType = std::array<uint16_t, LdmiRawFrameHeaderIn2352::countOfPowerModes>;

    //========================================
    //! \brief Size of whole message.
    //----------------------------------------
    static constexpr std::size_t payloadSize{4750U};

    //========================================
    //! \brief Offset where the reference id is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadIdOffset{192U};

    //========================================
    //! \brief Offset where the 'Number of row messages' (not number of rows)
    //! by normal power mode is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadNbOfRowsByNormalPowerOffset{2578U + 2U};

    //========================================
    //! \brief Offset where the 'Number of row messages' (not number of rows)
    //! by normal low mode is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadNbOfRowsByLowPowerOffset{2578U + 724U + 2U};

    //========================================
    //! \brief Offset where the 'Number of row messages' (not number of rows)
    //! by high power mode is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadNbOfRowsByHighPowerOffset{2578U + 724U + 724U + 2U};

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::LdmiRawStaticInfoIn2352";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    LdmiRawStaticInfoIn2352();

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another instance of frame static info.
    //----------------------------------------
    LdmiRawStaticInfoIn2352(LdmiRawStaticInfoIn2352&& other);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another instance of frame static info.
    //----------------------------------------
    LdmiRawStaticInfoIn2352(const LdmiRawStaticInfoIn2352& other);

    //========================================
    //! \brief Construct instance by message payload of LdmiA200 and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA200 message payload.
    //----------------------------------------
    explicit LdmiRawStaticInfoIn2352(const SharedBuffer& messagePayload);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LdmiRawStaticInfoIn2352() = default;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Another instance of frame static info.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawStaticInfoIn2352& operator=(LdmiRawStaticInfoIn2352&& other);

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Another instance of frame static info.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawStaticInfoIn2352& operator=(const LdmiRawStaticInfoIn2352& other);

public:
    //========================================
    //! \brief Compares two ldmi raw static infos for equality.
    //! \param[in] lhs  Ldmi raw static info in 2352.
    //! \param[in] rhs  Ldmi raw static info in 2352.
    //! \return Either \c true if both static infos are equal or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const LdmiRawStaticInfoIn2352& lhs, const LdmiRawStaticInfoIn2352& rhs);

    //========================================
    //! \brief Compares two ldmi raw static infos for inequality.
    //! \param[in] lhs  Ldmi raw static info in 2352.
    //! \param[in] rhs  Ldmi raw static info in 2352.
    //! \return Either \c true if both static infos are unequal or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const LdmiRawStaticInfoIn2352& lhs, const LdmiRawStaticInfoIn2352& rhs);

public: // getter
    //========================================
    //! \brief Get LdmiA200 message payload.
    //!
    //! Serialized form of static info which contains frame id and more.
    //!
    //! \return LdmiA200 message payload.
    //----------------------------------------
    const SharedBuffer& getMessagePayload() const;

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
    //! \brief Set LdmiA200 message payload and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA200 message payload.
    //----------------------------------------
    void setMessagePayload(const SharedBuffer& messagePayload);

private:
    //========================================
    //! \brief Message payload.
    //----------------------------------------
    SharedBuffer m_messagePayload;

    //========================================
    //! \brief Refernce id of static info.
    //----------------------------------------
    uint64_t m_id;

    //========================================
    //! \brief Number of rows by frame power mode.
    //----------------------------------------
    FrameConfigByModeType m_nbOfRows;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
