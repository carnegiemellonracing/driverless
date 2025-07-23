//==============================================================================
//!\file
//!
//! \brief Definition of a Some/IP data package.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 23rd, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/NetworkDataPackage.hpp>

#if WIN32
#    pragma warning(disable : 4293) // to avoid warning on windows for E2EProfile7CrcCalculatorType
#    include <boost/crc.hpp>
#    pragma warning(default : 4293)
#else
#    include <boost/crc.hpp>
#endif

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package to store Some/IP messages.
//------------------------------------------------------------------------------
class SomeIpDataPackage : public common::sdk::NetworkDataPackage
{
public:
    //========================================
    //! \brief Some/IP protocol version.
    //----------------------------------------
    static constexpr uint8_t protocolVersion{1};

    //========================================
    //! \brief Some/IP interface version.
    //----------------------------------------
    static constexpr uint8_t interfaceVersion{1};

    //========================================
    //! \brief SOME/IP service channel configuration.
    //----------------------------------------
    struct ChannelConfig
    {
        uint16_t serviceId;
        uint16_t methodId;
        bool e2eEnabled;
        uint32_t e2eDataId;
    };

    //========================================
    //! \brief Size of SOME/IP base header.
    //----------------------------------------
    static constexpr std::size_t sizeOfSomeIpBaseHeader{8};

    //========================================
    //! \brief Size of SOME/IP header.
    //----------------------------------------
    static constexpr std::size_t sizeOfSomeIpHeader{16};

    //========================================
    //! \brief Size of SOME/IP E2E crc.
    //----------------------------------------
    static constexpr std::size_t sizeOfE2EChecksum{8};

    //========================================
    //! \brief Size of SOME/IP E2E header.
    //----------------------------------------
    static constexpr std::size_t sizeOfE2EHeader{20};

    //========================================
    //! \brief SOME/IP E2E Profile7 checksum calculator type.
    //----------------------------------------
    using E2EProfile7CrcCalculatorType
        = boost::crc_optimal<64, 0x42F0E1EBA9EA3693, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, true, true>;

    //========================================
    //! \brief Some/IP message type enum.
    //----------------------------------------
    enum class MessageType : uint8_t
    {
        Request               = 0x00,
        RequestWithNoReturn   = 0x01,
        Notification          = 0x02,
        Response              = 0x80,
        Error                 = 0x81,
        TpRequest             = 0x20,
        TpRequestWithNoReturn = 0x21,
        TpNotification        = 0x22,
        TpResponse            = 0x23,
        TpError               = 0x24,
    };

    //========================================
    //! \brief Some/IP return code enum.
    //----------------------------------------
    enum class ReturnCode : uint8_t
    {
        Ok                    = 0x00,
        NotOk                 = 0x01,
        UnknownService        = 0x02,
        UnknownMethod         = 0x03,
        NotReady              = 0x04,
        NotReachable          = 0x05,
        Timeout               = 0x06,
        WrongProtocolVersion  = 0x07,
        WrongInterfaceVersion = 0x08,
        MalformedMessage      = 0x09,
        WrongMessageType      = 0x0a,
    };

public:
    //========================================
    //! \brief Get next session id.
    //! \note Every call increase the session id.
    //! \return Next session id.
    //----------------------------------------
    static uint16_t getNextSessionId();

public:
    //========================================
    //! \brief Construct Some/IP message.
    //! \param[in] index              Index of package over all streamed.
    //! \param[in] serviceId          Some/IP service id.
    //! \param[in] serviceInstanceId  Some/IP service instance id.
    //! \param[in] requestMethodId    Some/IP request method id.
    //! \param[in] sessionId          (Optional) Some/IP request/response session id (default = SomeIpDataPackage::getNextSessionId()).
    //! \param[in] messageType        (Optional) Some/IP message type (default = Request).
    //! \param[in] returnCode         (Optional) Some/IP return code (default = Ok).
    //----------------------------------------
    SomeIpDataPackage(const uint32_t index,
                      const uint16_t serviceId,
                      const uint16_t serviceInstanceId,
                      const uint16_t requestMethodId,
                      const uint16_t sessionId      = SomeIpDataPackage::getNextSessionId(),
                      const MessageType messageType = MessageType::Request,
                      const ReturnCode returnCode   = ReturnCode::Ok);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SomeIpDataPackage() override;

public: // Some/IP request access
    //========================================
    //! \brief Get the Some/IP service id for this package.
    //! \returns The service id this package is originated from.
    //----------------------------------------
    uint16_t getServiceId() const;

    //========================================
    //! \brief Get the Some/IP service instance id for this package.
    //! \returns The service instance id this package is originated from.
    //----------------------------------------
    uint16_t getServiceInstanceId() const;

    //========================================
    //! \brief Get the Some/IP request method id of the request this package is from.
    //! \returns The method id of the request this package is the answer to.
    //----------------------------------------
    uint16_t getRequestMethodId() const;

    //========================================
    //! \brief Get the Some/IP request/response session id of the communication thread.
    //! \returns The request/response session id of the communication thread.
    //----------------------------------------
    uint16_t getSessionId() const;

    //========================================
    //! \brief Get the Some/IP message type for this package.
    //! \returns The message type of this package.
    //----------------------------------------
    MessageType getMessageType() const;

    //========================================
    //! \brief Get the Some/IP return code of the response package.
    //! \returns The return code of the response package.
    //----------------------------------------
    ReturnCode getReturnCode() const;

public:
    //========================================
    //! \brief Compare data packages for equality.
    //! \param[in] lhs  SomeIpDataPackage to compare.
    //! \param[in] rhs  SomeIpDataPackage to compare.
    //! \returns Either \c true if equal or otherwise \c false.
    //! \note This compares simply the payload.
    //----------------------------------------
    friend bool operator==(const SomeIpDataPackage& lhs, const SomeIpDataPackage& rhs);

    //========================================
    //! \brief Compare data packages for inequality.
    //! \param[in] lhs  SomeIpDataPackage to compare.
    //! \param[in] rhs  SomeIpDataPackage to compare.
    //! \returns Either \c true if not equal or otherwise \c false.
    //! \note This compares simply the payload.
    //----------------------------------------
    friend bool operator!=(const SomeIpDataPackage& lhs, const SomeIpDataPackage& rhs);

private:
    //========================================
    //! \brief Some/IP service id.
    //----------------------------------------
    uint16_t m_serviceId;

    //========================================
    //! \brief Some/IP service instance id.
    //----------------------------------------
    uint16_t m_serviceInstanceId;

    //========================================
    //! \brief Some/IP request method id.
    //----------------------------------------
    uint16_t m_requestMethodId;

    //========================================
    //! \brief Some/IP request/response session id of the communication thread.
    //----------------------------------------
    uint16_t m_sessionId;

    //========================================
    //! \brief Some/IP message type.
    //----------------------------------------
    MessageType m_messageType;

    //========================================
    //! \brief Some/IP return code.
    //----------------------------------------
    ReturnCode m_returnCode;
}; // class SomeIpDataPackage

//==============================================================================
//! \brief Nullable SomeIpDataPackage pointer.
//------------------------------------------------------------------------------
using SomeIpDataPackagePtr = std::shared_ptr<SomeIpDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
