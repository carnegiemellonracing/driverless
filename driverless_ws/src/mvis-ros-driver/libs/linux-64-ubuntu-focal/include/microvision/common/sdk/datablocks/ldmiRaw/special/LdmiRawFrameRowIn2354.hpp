//==============================================================================
//! \file
//!
//! \brief Definition of row data for raw ldmi frame 2354.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 6th, 2022
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
//! \brief Ldmi frame row helper for reassembling ldmi raw frame package.
//------------------------------------------------------------------------------
class LdmiRawFrameRowIn2354 final
{
public:
    //========================================
    //! \brief Maximal number of rows supported by ldmi.
    //----------------------------------------
    static constexpr std::size_t maxNumberOfRows{100U};

public:
    //========================================
    //! \brief Size of whole message.
    //----------------------------------------
    static constexpr std::size_t payloadSize{2320U};

    //========================================
    //! \brief Offset where the row id is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadRowIdOffset{1U};

    //========================================
    //! \brief Offset where the frame id is located in payload.
    //----------------------------------------
    static constexpr std::size_t payloadFrameIdOffset{4U};

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::LdmiRawFrameRowIn2354";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    LdmiRawFrameRowIn2354();

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another instance of frame row.
    //----------------------------------------
    LdmiRawFrameRowIn2354(LdmiRawFrameRowIn2354&& other);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another instance of frame row.
    //----------------------------------------
    LdmiRawFrameRowIn2354(const LdmiRawFrameRowIn2354& other);

    //========================================
    //! \brief Construct instance by message payload of LdmiA2F1 and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA2F1 message payload.
    //----------------------------------------
    explicit LdmiRawFrameRowIn2354(const SharedBuffer& messagePayload);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LdmiRawFrameRowIn2354() = default;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Another instance of frame row.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawFrameRowIn2354& operator=(LdmiRawFrameRowIn2354&& other);

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Another instance of frame row.
    //! \return Reference of this.
    //----------------------------------------
    LdmiRawFrameRowIn2354& operator=(const LdmiRawFrameRowIn2354& other);

public:
    //========================================
    //! \brief Compares two ldmi raw frame rows for equality.
    //! \param[in] lhs  Ldmi raw frame row in 2354.
    //! \param[in] rhs  Ldmi raw frame row in 2354.
    //! \return Either \c true if both frame rows are equal or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const LdmiRawFrameRowIn2354& lhs, const LdmiRawFrameRowIn2354& rhs);

    //========================================
    //! \brief Compares two ldmi raw frame rows for inequality.
    //! \param[in] lhs  Ldmi raw frame row in 2354.
    //! \param[in] rhs  Ldmi raw frame row in 2354.
    //! \return Either \c true if both frame rows are unequal or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const LdmiRawFrameRowIn2354& lhs, const LdmiRawFrameRowIn2354& rhs);

public: // getter
    //========================================
    //! \brief Get LdmiA2F1 message payload.
    //!
    //! Serialized form of frame row which cotains frame id and more.
    //!
    //! \return LdmiA2F1 message payload.
    //----------------------------------------
    const SharedBuffer& getMessagePayload() const;

    //========================================
    //! \brief Get row id.
    //! \return Row id.
    //----------------------------------------
    uint8_t getId() const;

    //========================================
    //! \brief Get frame id.
    //! \return Frame id.
    //----------------------------------------
    uint32_t getFrameId() const;

public: // setter
    //========================================
    //! \brief Set LdmiA2F1 message payload and deserialize information for reassembling.
    //! \param[in] messagePayload  LdmiA2F1 message payload.
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

    //========================================
    //! \brief Row id.
    //----------------------------------------
    uint8_t m_id;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
