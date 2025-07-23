//==============================================================================
//! \file
//!
//! \brief Receive data from UDP network channel.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 30, 2019
//------------------------------------------------------------------------------

#pragma once
//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataPackageNetworkReceiver.hpp>
#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/io/NetworkDataPackage.hpp>
#include <microvision/common/sdk/io/udp/UdpBase.hpp>

#include <microvision/common/logging/logging.hpp>

#include <boost/asio.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

#if defined _WIN32
#    pragma warning(push)
#    pragma warning(disable : 4250)
#endif

//==============================================================================
//! \brief Receive data from UDP network channel.
//! \extends UdpBase
//! \extends DataPackageNetworkReceiver
//------------------------------------------------------------------------------
class UdpReceiver : public UdpBase, public DataPackageNetworkReceiver
{
private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::UdpReceiver";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

    //========================================
    //! \brief Number of ASIO job status checks.
    //----------------------------------------
    static constexpr uint8_t m_nbOfJobStatusCheck{3};

public:
    //========================================
    //! \brief Receive buffer type.
    //----------------------------------------
    using BufferType = std::array<char, maxUdpDatagramSize>;

    //========================================
    //! \brief Receive queue type.
    //! \note Used the list implementation in way of the queue concept.
    //----------------------------------------
    using ReceiveQueueType = QueueType;

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
    UdpReceiver();

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    UdpReceiver(UdpReceiver&& toMove) = delete;

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    UdpReceiver(const UdpReceiver& toCopy) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~UdpReceiver() override;

public:
    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    UdpReceiver& operator=(UdpReceiver&& toMove) = delete;

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    UdpReceiver& operator=(const UdpReceiver& toCopy) = delete;

public: // implements DataPackageNetworkReceiver
    //========================================
    //! \brief Get ThreadSafe::Access to the input queue where all received data will store.
    //! \returns ThreadSafe::Access to the input queue.
    //----------------------------------------
    ThreadSafe<QueueType>::Access getInputQueue() override;

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
    //! \brief Receive thread main method that will run on connect.
    //! \param[in] worker  Execution background worker.
    //! \returns Either \c true if keep thread alive or otherwise \c false.
    //----------------------------------------
    bool receiveMain(BackgroundWorker& worker);

    //========================================
    //! \brief Listen without blocking on UDP message.
    //----------------------------------------
    void asyncReceive();

    //========================================
    //! \brief Receive handler of UDP message.
    //! \param[in] error            Boost error.
    //! \param[in] bytesReceived    Udp package size.
    //----------------------------------------
    void receiveHandler(const boost::system::error_code& error, const std::size_t bytesReceived);

    //========================================
    //! \brief Listen without blocking on socket ready signal.
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
    //! \brief Set receive buffer size on socket.
    //! \param[in] bufferSize  Size of buffer.
    //! \returns Either \c true if buffer size successful changed, otherwise \c false.
    //----------------------------------------
    bool updateReceiveBufferSize(const uint32_t bufferSize);

    //========================================
    //! \brief Make boost IP address from sdk IP address.
    //! \param[in] ipAddress  Sdk IP address
    //! \returns Boost IP address
    //----------------------------------------
    boost::asio::ip::address getAddress(const IpAddressPtr& ipAddress) const;

    //========================================
    //! \brief Set boost error code as caught system exception.
    //! \param[in] error    Boost error code.
    //! \param[in] message  Error message.
    //----------------------------------------
    void setLastBoostError(const boost::system::error_code& error, const std::string& message);

private:
    //========================================
    //! \brief Cache entry for boost ip address to string.
    //----------------------------------------
    struct EndpointIPAddressWithString
    {
        //========================================
        //! \brief Last boost UDP remote endpoint of received message.
        //----------------------------------------
        boost::asio::ip::address endpointAddress;

        //========================================
        //! \brief IP address of Last boost UDP remote endpoint of received message.
        //----------------------------------------
        std::string endpointAddressString;
    };

private:
    //========================================
    //! \brief Boost io service.
    //----------------------------------------
    boost::asio::io_service m_service;

    //========================================
    //! \brief Active configuration while connected.
    //----------------------------------------
    UdpConfigurationPtr m_activeConfiguration;

    //========================================
    //! \brief Count of received packages to use as package index.
    //----------------------------------------
    int64_t m_packageIndex;

    //========================================
    //! \brief Input buffer to store received bytes.
    //----------------------------------------
    BufferType m_inputBuffer;

    //========================================
    //! \brief Receive input queue of UDP data packages.
    //----------------------------------------
    ThreadSafe<ReceiveQueueType> m_receiveQueue;

    //========================================
    //! \brief Boost UDP socket.
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
    //! \brief Receive worker.
    //----------------------------------------
    BackgroundWorker m_receiveWorker;

    //========================================
    //! \brief Destination uri, created by active configuration.
    //----------------------------------------
    Uri m_destinationUri;

    //========================================
    //! \brief Last boost UDP remote endpoint of received message.
    //----------------------------------------
    EndpointIPAddressWithString m_storedLastEndpoint;
}; // class UdpReceiver

#if defined _WIN32
#    pragma warning(pop)
#endif // _WIN32

//==============================================================================
//! \brief Nullable UdpReceiver pointer.
//------------------------------------------------------------------------------
using UdpReceiverPtr = std::unique_ptr<UdpReceiver>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
