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

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Ldmi frame footer helper for reassembling ldmi raw frame package.
//------------------------------------------------------------------------------
class LdmiRawFrameFooterIn2352 final
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

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::LdmiRawFrameFooterIn2352";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    LdmiRawFrameFooterIn2352();

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another instance of frame footer.
    //----------------------------------------
    LdmiRawFrameFooterIn2352(LdmiRawFrameFooterIn2352&& other);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another instance of frame footer.
    //----------------------------------------
    LdmiRawFrameFooterIn2352(const LdmiRawFrameFooterIn2352& other);

    //========================================
    //! \brief Construct instance by message payload of LdmiA2F2 and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA2F2 message payload.
    //----------------------------------------
    explicit LdmiRawFrameFooterIn2352(const SharedBuffer& messagePayload);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LdmiRawFrameFooterIn2352() = default;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Another instance of frame footer.
    //! \returns Reference of this.
    //----------------------------------------
    LdmiRawFrameFooterIn2352& operator=(LdmiRawFrameFooterIn2352&& other);

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Another instance of frame footer.
    //! \returns Reference of this.
    //----------------------------------------
    LdmiRawFrameFooterIn2352& operator=(const LdmiRawFrameFooterIn2352& other);

public:
    //========================================
    //! \brief Compares two ldmi raw frame footers for equality.
    //! \param[in] lhs  Ldmi raw frame footer in 2352.
    //! \param[in] rhs  Ldmi raw frame footer in 2352.
    //! \returns Either \c true if both frame footers are equal or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const LdmiRawFrameFooterIn2352& lhs, const LdmiRawFrameFooterIn2352& rhs);

    //========================================
    //! \brief Compares two ldmi raw frame footers for inequality.
    //! \param[in] lhs  Ldmi raw frame footer in 2352.
    //! \param[in] rhs  Ldmi raw frame footer in 2352.
    //! \returns Either \c true if both frame footers are unequal or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const LdmiRawFrameFooterIn2352& lhs, const LdmiRawFrameFooterIn2352& rhs);

public: // getter
    //========================================
    //! \brief Get LdmiA2F2 message payload.
    //!
    //! Serialized form of frame footer which contains frame id and more.
    //!
    //! \returns LdmiA2F2 message payload.
    //----------------------------------------
    const SharedBuffer& getMessagePayload() const;

    //========================================
    //! \brief Get frame id.
    //! \returns Frame id.
    //----------------------------------------
    uint32_t getFrameId() const;

public: // setter
    //========================================
    //! \brief Set LdmiA2F2 message payload and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA2F2 message payload.
    //----------------------------------------
    void setMessagePayload(const SharedBuffer& messagePayload);

private:
    //========================================
    //! \brief Message payload.
    //----------------------------------------
    SharedBuffer m_messagePayload;

    //========================================
    //! \brief Frame id.
    //----------------------------------------
    uint32_t m_frameId;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
