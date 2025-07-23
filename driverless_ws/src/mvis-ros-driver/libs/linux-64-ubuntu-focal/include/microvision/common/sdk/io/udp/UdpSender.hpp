//==============================================================================
//! \file
//!
//! \brief Send data via UDP network protocol.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Okt 2, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataPackageNetworkSender.hpp>
#include <microvision/common/sdk/io/NetworkDataPackage.hpp>
#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/misc/SharedBuffer.hpp>
#include <microvision/common/sdk/io/udp/UdpBase.hpp>

#include <microvision/common/logging/logging.hpp>

#include <boost/asio.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Send data via UDP network protocol.
//! \extends UdpBase
//! \extends DataPackageNetworkSender
//------------------------------------------------------------------------------
class UdpSender : public UdpBase, public DataPackageNetworkSender
{
private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::UdpSender";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Output queue type.
    //----------------------------------------
    using SendQueueType = QueueType;

    //========================================
    //! \brief Network interface type name.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

public:
    //========================================
    //! \brief Get name of type of this network interface.
    //! \returns NetworkInterface type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getTypeName();

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    UdpSender();

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    UdpSender(UdpSender&& toMove) = delete;

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    UdpSender(const UdpSender& toCopy) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~UdpSender() override;

public:
    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    UdpSender& operator=(UdpSender&& toMove) = delete;

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    UdpSender& operator=(const UdpSender& toCopy) = delete;

public: // implements DataPackageNetworkReceiver
    //========================================
    //! \brief Get ThreadSafe::Access to the output queue where all send-able data will store.
    //! \returns ThreadSafe::Access to the output queue.
    //----------------------------------------
    ThreadSafe<QueueType>::Access getOutputQueue() override;

public: // implements NetworkBase
    //========================================
    //! \brief Get the source/destination Uri.
    //! \return Describing source/destination Uri of stream.
    //----------------------------------------
    Uri getUri() const override;

    //========================================
    //! \brief Checks that a connection is established.
    //! \returns Either \c true if connection is established or otherwise \c false.
    //----------------------------------------
    bool isConnected() const override;

    //========================================
    //! \brief Checks that a connection is established and packages are still processed.
    //! \returns Either \c true if connection is established or work is ongoing on received data or otherwise \c false.
    //----------------------------------------
    bool isWorking() override;

    //========================================
    //! \brief Get the last error which is caught in network io thread.
    //! \returns Exception pointer if error caught, otherwise nullptr.
    //----------------------------------------
    std::exception_ptr getLastError() const override;

    //========================================
    //! \brief Establish a connection to the network resource.
    //----------------------------------------
    void connect() override;

    //========================================
    //! \brief Disconnect from the network resource.
    //----------------------------------------
    void disconnect() override;

public: // implements NetworkInterface
    //========================================
    //! \brief Get network interface type which is used to identify the implementation.
    //!
    //! Network interface type is a human readable unique string name of the
    //! NetworkInterface used to address it in code.
    //!
    //! \returns Human readable unique name of network interface implementation.
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get all errors which are caught in network io threads.
    //! \returns List of exception pointers.
    //----------------------------------------
    std::vector<std::exception_ptr> getErrors() const override;

private:
    //========================================
    //! \brief Disconnect from the network resource without join io thread.
    //----------------------------------------
    void internalDisconnect();

    //========================================
    //! \brief Send thread main method that will run on connect.
    //----------------------------------------
    bool sendMain(BackgroundWorker& worker);

    //========================================
    //! \brief Send async the next UDP message.
    //----------------------------------------
    void asyncSend();

    //========================================
    //! \brief Send handler of UDP message.
    //! \param[in] error            Boost error.
    //! \param[in] bytesWritten     Sent bytes.
    //----------------------------------------
    void sendHandler(const boost::system::error_code& error, const std::size_t bytesWritten);

    //========================================
    //! \brief Listen async on socket ready signal.
    //----------------------------------------
    void asyncWait();

    //========================================
    //! \brief Ready signal handler of UDP socket.
    //! \param[in] error  Boost error.
    //----------------------------------------
    void waitHandler(const boost::system::error_code& error);

    //========================================
    //! \brief Start async timeout timer.
    //----------------------------------------
    void asyncTimeout();

    //========================================
    //! \brief Handle timeout.
    //! \param[in] error  Boost error.
    //----------------------------------------
    void timeoutHandler(const boost::system::error_code& error);

    //========================================
    //! \brief Set send buffer size on socket.
    //! \param[in] bufferSize  Size of buffer.
    //! \returns Either \c true if buffer size successful changed, otherwise \c false.
    //----------------------------------------
    bool updateSendBufferSize(const uint32_t bufferSize);

    //========================================
    //! \brief Make boost IP address from sdk IP address.
    //! \param[in] ipAddress  Sdk IP address
    //! \returns Boost IP address
    //----------------------------------------
    boost::asio::ip::address getAddress(const IpAddressPtr& ipAddress) const;

private:
    //========================================
    //! \brief Active configuration while connected.
    //----------------------------------------
    UdpConfigurationPtr m_activeConfiguration;

    //========================================
    //! \brief Current package to send via UDP messages.
    //----------------------------------------
    DataPackagePtr m_currentOutputPackage;

    //========================================
    //! \brief Sended bytes of current package.
    //----------------------------------------
    std::size_t m_currentOutputMessageWrittenBytes;

    //========================================
    //! \brief Boost io service.
    //----------------------------------------
    boost::asio::io_service m_service;

    //========================================
    //! \brief Boost udp socket.
    //----------------------------------------
    boost::asio::ip::udp::socket m_socket;

    //========================================
    //! \brief Boost timeout timer.
    //----------------------------------------
    boost::asio::deadline_timer m_timeoutTimer;

    //========================================
    //! \brief Boost UDP remote endpoint of received message.
    //----------------------------------------
    boost::asio::ip::udp::endpoint m_remoteEndpoint;

    //========================================
    //! \brief Send queue of UDP data packages.
    //----------------------------------------
    ThreadSafe<SendQueueType> m_sendQueue;

    //========================================
    //! \brief Send worker.
    //----------------------------------------
    BackgroundWorker m_sendWorker;
}; // class UdpSender

//==============================================================================
//! \brief Nullable UdpSender pointer.
//------------------------------------------------------------------------------
using UdpSenderPtr = std::unique_ptr<UdpSender>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
