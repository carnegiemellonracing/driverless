//==============================================================================
//! \file
//!
//! \brief Interface for TCP io network resources.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 24, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/defines/boost.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/sdk/config/io/TcpConfiguration.hpp>
#include <microvision/common/sdk/io/NetworkInterface.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface for TCP io network resources.
//!
//! Implement this interface to provide TCP network resource handling.
//!
//! \extends NetworkInterface
//------------------------------------------------------------------------------
class TcpBase : public NetworkInterface
{
public:
    //========================================
    //! \brief TCP socket type.
    //----------------------------------------
    using BoostSocketType = typename boost::asio::ip::tcp::socket;

    //========================================
    //! \brief TCP endpoint type.
    //----------------------------------------
    using BoostEndpointType = typename boost::asio::ip::tcp::endpoint;

    //========================================
    //! \brief Maximum size of a TCP datagram.
    //!
    //! 65.535 max datagram size - 20 Byte TCP Header - 20 Byte IPv4 Header
    //----------------------------------------
    static constexpr uint16_t maxTcpDatagramSize{65495U};

    //========================================
    //! \brief Timer type.
    //----------------------------------------
    using BoostTimerType = boost::asio::deadline_timer;

    //========================================
    //! \brief Number of ASIO job status checks.
    //----------------------------------------
    static constexpr uint8_t nbOfJobStatusCheckRetries{3};

protected:
    //========================================
    //! \brief Callback interface for error handling.
    //----------------------------------------
    class ErrorSupport
    {
    public:
        //========================================
        //! \brief This function will be called when an error has been caught.
        //! \param[in] exception  Pointer to caught exception.
        //----------------------------------------
        virtual void checkForError(const std::exception_ptr& exception) = 0;

        //========================================
        //! \brief This function will be called when an error has been caught.
        //! \param[in] errorCode    Error code of system see \a std::errc enum.
        //! \param[in] message      Error message with a hint or about the reason.
        //----------------------------------------
        virtual void checkForError(const int errorCode, const std::string& message) = 0;

    }; // class ErrorSupport

    //========================================
    //! \brief Abstract base class to provide receive/send functionality.
    //! \extends ErrorSupport
    //----------------------------------------
    class ConnectionSupport : public virtual ErrorSupport
    {
    public:
        //========================================
        //! \brief Constructor with required boost io context reference.
        //! \param[in] service  Boost io context for async job handling.
        //----------------------------------------
        explicit ConnectionSupport(BoostIoContext& service);

        //========================================
        //! \brief Default destructor.
        //----------------------------------------
        ~ConnectionSupport();

    public:
        //========================================
        //! \brief Get socket for connection/acception.
        //! \return Boost socket instance.
        //----------------------------------------
        BoostSocketType& getSocket();

        //========================================
        //! \brief Get socket for connection/accept.
        //! \return Boost socket instance.
        //----------------------------------------
        const BoostSocketType& getSocket() const;

        //========================================
        //! \brief Set remote/local uri.
        //----------------------------------------
        void init();

        //========================================
        //! \brief Listen without blocking on receiving TCP messages. Then \a onDataReceived will be called.
        //----------------------------------------
        void receiveData();

        //========================================
        //! \brief Send data package until all bytes are written. Then \a onDataSent will be called.
        //! \param[in] package  The data package which has to be sent.
        //----------------------------------------
        void sendData(const NetworkDataPackagePtr& package);

        //========================================
        //! \brief Disconnect connection and after that the \a onDisconnect will called.
        //----------------------------------------
        void disconnect();

    public:
        //========================================
        //! \brief Set receive buffer size on socket.
        //! \param[in] bufferSize  Size of buffer.
        //! \returns Either \c true if buffer size successful changed, otherwise \c false.
        //----------------------------------------
        bool updateReceiveBufferSize(const uint32_t bufferSize);

        //========================================
        //! \brief Set send buffer size on socket.
        //! \param[in] bufferSize  Size of buffer.
        //! \returns Either \c true if buffer size successful changed, otherwise \c false.
        //----------------------------------------
        bool updateSendBufferSize(const uint32_t bufferSize);

    public:
        //========================================
        //! \brief Will be called when data package is received.
        //! \param[in] package  Received data package.
        //----------------------------------------
        virtual void onDataReceived(const NetworkDataPackagePtr& package) = 0;

        //========================================
        //! \brief Will be called when data package completely sent.
        //! \param[in] inIdle  Either \c true if data queue is empty, otherwise \c false.
        //----------------------------------------
        virtual void onDataSent(const bool inIdle) = 0;

        //========================================
        //! \brief Will be called when connection disconnected.
        //----------------------------------------
        virtual void onDisconnect() = 0;

        //========================================
        //! \brief Check wether the connection is established.
        //! \return Either \c true if connection is established, otherwise \c false.
        //----------------------------------------
        virtual bool isConnected() const = 0;

    private:
        //========================================
        //! \brief Listen without blocking on receiving TCP message.
        //----------------------------------------
        void asyncReceive();

        //========================================
        //! \brief Receive handler of TCP message.
        //! \param[in] error            Boost error.
        //! \param[in] bytesReceived    TCP package size.
        //----------------------------------------
        void receiveHandler(const boost::system::error_code& error, const std::size_t bytesReceived);

        //========================================
        //! \brief Listen without blocking on socket ready signal.
        //----------------------------------------
        void asyncReceiveWait();

        //========================================
        //! \brief Ready signal handler of TCP socket.
        //! \param[in] error  Boost error.
        //----------------------------------------
        void waitReceiveHandler(const boost::system::error_code& error);

        //========================================
        //! \brief Send the current data package as TCP message asynchronously.
        //----------------------------------------
        void asyncSend();

        //========================================
        //! \brief Send handler of TCP message.
        //! \param[in] error            Boost error.
        //! \param[in] bytesWritten     Sent bytes.
        //----------------------------------------
        void sendHandler(const boost::system::error_code& error, const std::size_t bytesWritten);

        //========================================
        //! \brief Listen without blocking on socket ready signal.
        //----------------------------------------
        void asyncSendWait();

        //========================================
        //! \brief Ready signal handler of TCP socket.
        //! \param[in] error  Boost error.
        //----------------------------------------
        void waitSendHandler(const boost::system::error_code& error);

    private:
        //========================================
        //! \brief Boost TCP socket.
        //----------------------------------------
        BoostSocketType m_socket;

        //========================================
        //! \brief Remote uri, created by configuration.
        //----------------------------------------
        Uri m_remoteUri;

        //========================================
        //! \brief Local uri, created by socket.
        //----------------------------------------
        Uri m_localUri;

        //========================================
        //! \brief Count of received packages to use as package index.
        //----------------------------------------
        int64_t m_packageIndex;

        //========================================
        //! \brief Current package to be received via TCP messages.
        //----------------------------------------
        NetworkDataPackagePtr m_currentInputPackage;

        //========================================
        //! \brief Current package to be send via TCP messages.
        //----------------------------------------
        DataPackagePtr m_currentOutputPackage;

        //========================================
        //! \brief Already sent bytes of current package.
        //----------------------------------------
        std::size_t m_currentOutputMessageWrittenBytes;

        //========================================
        //! \brief Data output queue for sending data.
        //----------------------------------------
        ThreadSafe<std::list<NetworkDataPackagePtr>> m_outputQueue;

    }; // class ConnectionSupport

    //========================================
    //! \brief Abstract base class to provide timeout functionality.
    //! \extends ErrorSupport
    //----------------------------------------
    class TimeoutSupport : public virtual ErrorSupport
    {
    public:
        //========================================
        //! \brief Constructor with required boost io context reference.
        //! \param[in] service  Boost io context for async job handling.
        //----------------------------------------
        explicit TimeoutSupport(BoostIoContext& service);

        //========================================
        //! \brief Default destructor.
        //----------------------------------------
        ~TimeoutSupport();

    public:
        //========================================
        //! \brief Start/restart timer with stored timeout in ms.
        //! \details If timeout in ms is \c 0 it will not be started.
        //----------------------------------------
        void reset();

        //========================================
        //! \brief Set stored timeout in ms and start/restart timer.
        //! \details If timeout in ms is \c 0 it will not be started/restarted.
        //! \param[in] timeoutInMs  Timeout in milliseconds.
        //----------------------------------------
        void reset(uint32_t timeoutInMs);

        //========================================
        //! \brief Stop timeout timer.
        //----------------------------------------
        void cancel();

    public:
        //========================================
        //! \brief Will be called when the timeout occurred.
        //----------------------------------------
        virtual void onTimeoutOccurred() = 0;

        //========================================
        //! \brief Will be called when the timer is cancelled.
        //----------------------------------------
        virtual void onCancel() = 0;

    private:
        //========================================
        //! \brief Start async timeout timer.
        //----------------------------------------
        void asyncTimeout();

        //========================================
        //! \brief Handle timeout.
        //! \param[in] error  Boost error.
        //----------------------------------------
        void timeoutHandler(const boost::system::error_code& error);

    public:
        //========================================
        //! \brief Timeout in milliseconds.
        //----------------------------------------
        uint32_t m_timeoutInMs;

        //========================================
        //! \brief Boost timeout timer.
        //----------------------------------------
        BoostTimerType m_timeoutTimer;

    }; // class TimeoutSupport

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::TcpBase";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    TcpBase();

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    TcpBase(TcpBase&& other) = delete;

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    TcpBase(const TcpBase& other) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~TcpBase() override = default;

public:
    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    TcpBase& operator=(TcpBase&& other) = delete;

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    TcpBase& operator=(const TcpBase& other) = delete;

public: // implements Configurable
    //========================================
    //! \brief Get supported types of configuration.
    //!
    //! Configuration type is a human readable unique string name of the configuration
    //! used to address it in code.
    //!
    //! \returns All supported configuration types.
    //----------------------------------------
    const std::vector<std::string>& getConfigurationTypes() const override;

public: // implements NetworkBase
    //========================================
    //! \brief Get the source/destination Uri.
    //! \return Describing source/destination Uri of stream.
    //----------------------------------------
    Uri getUri() const override;

public: // implements NetworkInterface
    //========================================
    //! \brief Get pointer to network configuration which is used to establishing the connection.
    //!
    //! The pointer points to the tcp network configuration.
    //!
    //! \return Pointer to an instance of NetworkConfiguration.
    //----------------------------------------
    NetworkConfigurationPtr getConfiguration() const override;

    //========================================
    //! \brief Set pointer to network configuration which is used to establishing the connection.
    //!
    //! The pointer points to the tcp network configuration.
    //!
    //! \param[in] configuration  Pointer to an instance of NetworkConfiguration.
    //! \return Either \c true if the configuration is supported by implementation or otherwise \c false.
    //! \note If the configuration is not supported by implementation it will not change the current value.
    //!       However, if \a configuration is \c nullptr the configuration of NetworkInterface will be reset.
    //----------------------------------------
    bool setConfiguration(const NetworkConfigurationPtr& configuration) override;

protected:
    //========================================
    //! \brief Boost io service.
    //----------------------------------------
    boost::asio::io_service m_service;

    //========================================
    //! \brief TCP configuration.
    //----------------------------------------
    TcpConfigurationPtr m_configuration;

}; // class TcpBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
