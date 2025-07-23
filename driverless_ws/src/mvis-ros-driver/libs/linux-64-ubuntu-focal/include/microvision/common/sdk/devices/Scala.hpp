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
//! \date Oct 04, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcEthDevice.hpp>
#include <microvision/common/sdk/devices/IdcEthType.hpp>

//==============================================================================

// Change the compiler warning settings until ALLOW_WARNINGS_END.
ALLOW_WARNINGS_BEGIN
// Allow deprecated warnings in deprecated code to avoid
// compiler errors because of deprecated dependencies.
ALLOW_WARNING_DEPRECATED

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\class Scala
//! \brief Class to connect to a Scala sensor.
//! \date Oct 1, 2013
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED Scala final : public IdcEthDevice
{
public:
    //========================================
    //!\brief Create a Scala (connection class).
    //!
    //! This constructor will create a Scala class object
    //! which will try to connect to a Scala sensor,
    //! using the given IP address and port number using TCP protocol.
    //!
    //! \param[in] ip      IP address of the scanner
    //!                    to be connected with.
    //! \param[in] port    Port number for the connection
    //!                    with the scanner.
    //! \param[in] ethTcp  Protocol type for the connection
    //!                    with the scanner.
    //----------------------------------------
    Scala(const std::string& ip, const unsigned short port = 12004, const IdcEthTypeTcp& ethTcp = IdcEthTypeTcp());

    //========================================
    //!\brief Create a Scala (connection class).
    //!
    //! This constructor will create a Scala class object
    //! which will try to connect to a Scala sensor,
    //! using the given IP address, port number and ethernet protocol.
    //!
    //! \param[in] ip      IP address of the scanner
    //!                    to be connected with.
    //! \param[in] port    Port number for the connection
    //!                    with the scanner.
    //! \param[in] ethTcp  Protocol type for the connection
    //!                    with the scanner.
    //! \param[in] ifa     Address of network interface for the connection
    //!                    with the scanner.
    //----------------------------------------
    Scala(const std::string& ip,
          const unsigned short port,
          const IdcEthTypeUdp& ethUdp,
          const boost::asio::ip::address_v4 ifa = boost::asio::ip::address_v4::any());

    //========================================
    //!\brief Create a Scala (connection class).
    //!
    //! This constructor will create a Scala class object
    //! which will try to connect to a Scala sensor,
    //! using the given IP address, port number and ethernet protocol.
    //!
    //! \param[in] ip      IP address of the scanner
    //!                    to be connected with.
    //! \param[in] port    Port number for the connection
    //!                    with the scanner.
    //! \param[in] ethTcp  Protocol type for the connection
    //!                    with the scanner.
    //! \param[in] ifa     Address of network interface for the connection
    //!                    with the scanner.
    //----------------------------------------
    Scala(const std::string& ip,
          const unsigned short port,
          const IdcEthTypeUdpMulticast& ethMulticast,
          const boost::asio::ip::address_v4 ifa = boost::asio::ip::address_v4::any());

    //========================================
    //!\brief Destructor.
    //!
    //! Will disconnect before destruction.
    //----------------------------------------
    virtual ~Scala();

public:
    void connect(const uint32_t timeoutSec = IdcEthDevice::defaultReceiveTimeoutSeconds) override;

private:
    enum class EthType : uint8_t
    {
        EthTcp,
        EthUdp,
        EthUdpMulticast
    };

private: // not supported
    using IdcEthDevice::sendCommand;

private:
    std::string m_strIP;
    unsigned short m_port{0};
    boost::asio::ip::address_v4 m_ifa;
    EthType m_ethType;
}; // Scala

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
