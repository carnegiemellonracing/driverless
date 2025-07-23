//==============================================================================
//! \file
//!
//! \brief TCP server network interface.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 3, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/tcp/TcpBase.hpp>
#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/io/DataPackageNetworkSender.hpp>
#include <microvision/common/sdk/io/DataPackageNetworkReceiver.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief TCP server network interface.
//! \extends TcpBase
//! \extends DataPackageNetworkSender
//! \extends DataPackageNetworkReceiver
//------------------------------------------------------------------------------
class TcpServer final : public TcpBase, public DataPackageNetworkSender, public DataPackageNetworkReceiver
{
public:
    using BoostTcpAcceptor = boost::asio::ip::tcp::acceptor;

public:
    //========================================
    //! \brief Number of threads which are instantiatet for asynchronously network communication.
    //----------------------------------------
    static constexpr uint8_t defaultNumberOfThreads{2};

    //========================================
    //! \brief Get name of type of this network interface.
    //! \returns NetworkInterface type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getTypeName();

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::TcpServer";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

    //========================================
    //! \brief Pre definition of Session for SessionHandler.
    //----------------------------------------
    class Session;

    //========================================
    //! \brief Abstract interface for session controller.
    //----------------------------------------
    class SessionHandler
    {
    public:
        virtual ~SessionHandler() = default;

    public: // implements session handling
        //========================================
        //! \brief This function will called when error caught.
        //! \param[in] exception  Pointer to caught exception.
        //----------------------------------------
        virtual void checkForErrorInSession(const Session& session, const std::exception_ptr& exception) = 0;

        //========================================
        //! \brief This function will called when error caught.
        //! \param[in] errorCode    Error code of system see \a std::errc enum.
        //! \param[in] message      Error message with a hint or about the reason.
        //----------------------------------------
        virtual void checkForErrorInSession(const Session& session, const int errorCode, const std::string& message)
            = 0;

        //========================================
        //! \brief Will be called when data package is received in session.
        //! \param[in] session  Session which the data has received.
        //! \param[in] package  Received data package.
        //----------------------------------------
        virtual void onDataReceivedInSession(const Session& session, const NetworkDataPackagePtr& package) = 0;

        //========================================
        //! \brief Will be called when data package complete sended in session.
        //! \param[in] session  Session which the data has sent.
        //----------------------------------------
        virtual void onDataSentInSession(const Session& session) = 0;

        //========================================
        //! \brief Will be called when session disconnected.
        //! \param[in] session  Session which has disconnected.
        //----------------------------------------
        virtual void onDisconnectInSession(const Session& session) = 0;

    }; // class SessionHandler

    //========================================
    //! \brief Implements Server <-> Client communication.
    //! \extends TcpBase::ConnectionSupport
    //! \extends TcpBase::TimeoutSupport
    //----------------------------------------
    class Session : public ConnectionSupport, public TimeoutSupport
    {
    public:
        //========================================
        //! \brief Constructor with parameters for io communication.
        //! \param[in] context                 Boost io context.
        //! \param[in] numberOfNetworkThreads  Number of threads for asynchronously network communication.
        //----------------------------------------
        Session(SessionHandler& handler, BoostIoContext& context);

        //========================================
        //! \brief Default destructor.
        //----------------------------------------
        virtual ~Session();

    public: // implements ErrorSupport
        //========================================
        //! \brief This function will called when error caught.
        //! \param[in] exception  Pointer to caught exception.
        //----------------------------------------
        void checkForError(const std::exception_ptr& exception) override;

        //========================================
        //! \brief This function will called when error caught.
        //! \param[in] errorCode    Error code of system see \a std::errc enum.
        //! \param[in] message      Error message with a hint or about the reason.
        //----------------------------------------
        void checkForError(const int errorCode, const std::string& message) override;

    public: // implements ConnectionSupport
        //========================================
        //! \brief Will be called when data package is received.
        //! \param[in] package  Received data package.
        //----------------------------------------
        void onDataReceived(const NetworkDataPackagePtr& package) override;

        //========================================
        //! \brief Will be called when data package complete sended.
        //! \param[in] inIdle  Either \c true if data queue is empty, otherwise \c false.
        //----------------------------------------
        void onDataSent(const bool inIdle) override;

        //========================================
        //! \brief Will be called when connection disconnected.
        //----------------------------------------
        void onDisconnect() override;

        //========================================
        //! \brief Check wether the connection is established.
        //! \return Either \c true if connection is established, otherwise \c false.
        //----------------------------------------
        bool isConnected() const override;

    public: // implements TimeoutSupport
        //========================================
        //! \brief Will called when the timeout occurred.
        //----------------------------------------
        void onTimeoutOccurred() override;

        //========================================
        //! \brief Will called when the timer is canceld.
        //----------------------------------------
        void onCancel() override;

    public:
        //========================================
        //! \brief Session handler to control all connections.
        //----------------------------------------
        SessionHandler& m_handler;

        //========================================
        //! \brief Session client endpoint.
        //----------------------------------------
        BoostEndpointType m_clientEndpoint;

        //========================================
        //! \brief Either \c true if session will disconnected externally, otherwise \c false.
        //----------------------------------------
        std::atomic_bool m_externalDisconnect;

    }; // class Session

    //========================================
    //! \brief Nullable pointer of Session instance.
    //----------------------------------------
    using SessionUPtr = std::unique_ptr<Session>;

    //========================================
    //! \brief Implements server and session handling.
    //! \extends SessionHandler
    //! \extends ErrorSupport
    //----------------------------------------
    class ServerImpl : public SessionHandler, public ErrorSupport
    {
    public:
        //========================================
        //! \brief Constructor with parameters for io communication.
        //! \param[in] context                 Boost io context.
        //! \param[in] numberOfNetworkThreads  Number of threads for asynchronously network communication.
        //----------------------------------------
        ServerImpl(BoostIoContext& context, uint8_t numberOfNetworkThreads);

        //========================================
        //! \brief Default destructor.
        //----------------------------------------
        ~ServerImpl();

    public:
        //========================================
        //! \brief Start up all threads for asynchronous network communication.
        //----------------------------------------
        void start();

        //========================================
        //! \brief Shutdown all threads for asynchronously network communication.
        //----------------------------------------
        void stop();

    public: // implements ErrorSupport
        //========================================
        //! \brief This function will be called when an error is caught.
        //! \param[in] exception  Pointer to caught exception.
        //----------------------------------------
        void checkForError(const std::exception_ptr& exception) override;

        //========================================
        //! \brief This function will be called when an error is caught.
        //! \param[in] errorCode    Error code of system see \a std::errc enum.
        //! \param[in] message      Error message with a hint or about the reason.
        //----------------------------------------
        void checkForError(const int errorCode, const std::string& message) override;

    public: // implements SessionHandler
        //========================================
        //! \brief This function will be called when an error is caught.
        //! \param[in] exception  Pointer to caught exception.
        //----------------------------------------
        void checkForErrorInSession(const Session& session, const std::exception_ptr& exception) override;

        //========================================
        //! \brief This function will be called when an error is caught.
        //! \param[in] errorCode    Error code of system see \a std::errc enum.
        //! \param[in] message      Error message with a hint or about the reason.
        //----------------------------------------
        void checkForErrorInSession(const Session& session, const int errorCode, const std::string& message) override;

        //========================================
        //! \brief Will be called when data package is received in session.
        //! \param[in] session  Session which the data has received.
        //! \param[in] package  Received data package.
        //----------------------------------------
        void onDataReceivedInSession(const Session& session, const NetworkDataPackagePtr& package) override;

        //========================================
        //! \brief Will be called when data package complete sended in session.
        //! \param[in] session  Session which the data has sent.
        //----------------------------------------
        void onDataSentInSession(const Session& session) override;

        //========================================
        //! \brief Will be called when session disconnected.
        //! \param[in] session  Session which has disconnected.
        //----------------------------------------
        void onDisconnectInSession(const Session& session) override;

    private:
        //========================================
        //! \brief Run io context to use this thread for receiving/sending data etc.
        //! \return Either \c true if socket is still open, otherwise \c false.
        //----------------------------------------
        bool run();

        //========================================
        //! \brief Establish connection without blocking against TCP server.
        //----------------------------------------
        void asyncAccept();

        //========================================
        //! \brief Handle result of connection esteblishing.
        //! \param[in] error  Boost error.
        //----------------------------------------
        void acceptHandler(const boost::system::error_code& error);

        //========================================
        //! \brief Output queue observer thread main.
        //! \param[in] worker  Backround worker instance.
        //----------------------------------------
        bool observeOutputQueue(BackgroundWorker& worker);

    public:
        //========================================
        //! \brief Boost io service.
        //----------------------------------------
        BoostIoContext& m_service;

        //========================================
        //! \brief Boost acceptor for connection handshake.
        //----------------------------------------
        BoostTcpAcceptor m_acceptor;

        //========================================
        //! \brief Acceptor peer endpoint.
        //----------------------------------------
        BoostEndpointType m_peerEndpoint;

        //========================================
        //! \brief Communication timeout after milliseconds.
        //----------------------------------------
        uint32_t m_communicationTimeout;

        //========================================
        //! \brief Output queue observer thread.
        //----------------------------------------
        BackgroundWorker m_outputQueueObserver;

        //========================================
        //! \brief Thread resources sync handler.
        //----------------------------------------
        ThreadSyncPtr m_syncHandle;

        //========================================
        //! \brief All threads for asynchronously network communication.
        //----------------------------------------
        std::vector<BackgroundWorkerUPtr> m_serverThreads;

        //========================================
        //! \brief Data output queue for sending data.
        //----------------------------------------
        ThreadSafe<DataPackageNetworkSender::QueueType> m_outputQueue;

        //========================================
        //! \brief Data input queue for receiving data.
        //----------------------------------------
        ThreadSafe<DataPackageNetworkReceiver::QueueType> m_inputQueue;

        //========================================
        //! \brief Session of current handshake.
        //----------------------------------------
        SessionUPtr m_sessionToAccept;

        //========================================
        //! \brief TCP server sessions for io communication indexed by remote endpoint.
        //----------------------------------------
        ThreadSafe<std::map<BoostEndpointType, SessionUPtr>> m_sessions;

    }; // class ServerImpl

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    TcpServer();

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    TcpServer(TcpServer&& other) = delete;

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    TcpServer(const TcpServer& other) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~TcpServer() override;

public:
    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    TcpServer& operator=(TcpServer&& other) = delete;

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    TcpServer& operator=(const TcpServer& other) = delete;

public: // implements NetworkBase
    //========================================
    //! \brief Get the source/destination Uri.
    //! \return Describing source/destination Uri of network.
    //----------------------------------------
    Uri getUri() const override;

    //========================================
    //! \brief Get the last error which is caught in network io thread.
    //! \returns Exception pointer if error caught, otherwise nullptr.
    //----------------------------------------
    std::exception_ptr getLastError() const override;

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

    //========================================
    //! \brief Get handle to sync thread resources of network interfaces.
    //! \return Either pointer to sync handle or \c nullptr if not synced.
    //----------------------------------------
    ThreadSyncPtr getSynchronizationHandle() const override;

    //========================================
    //! \brief Set handle to sync thread resources of network interfaces.
    //! \param[in] syncHandle  Pointer to sync handle to enable sync or \c nullptr to disable sync.
    //----------------------------------------
    void setSynchronizationHandle(const ThreadSyncPtr& syncHandle) override;

public: // implements DataPackageNetworkSender
    //========================================
    //! \brief Get ThreadSafe::Access to the output queue where all sendable data will store.
    //! \returns ThreadSafe::Access to the output queue.
    //----------------------------------------
    ThreadSafe<DataPackageNetworkSender::QueueType>::Access getOutputQueue() override;

public: // implements DataPackageNetworkReceiver
    //========================================
    //! \brief Get ThreadSafe::Access to the input queue where all received data will store.
    //! \returns ThreadSafe::Access to the input queue.
    //----------------------------------------
    ThreadSafe<DataPackageNetworkReceiver::QueueType>::Access getInputQueue() override;

private:
    //========================================
    //! \brief Boost io context.
    //----------------------------------------
    BoostIoContext m_context;

    //========================================
    //! \brief Server implementation.
    //----------------------------------------
    ServerImpl m_server;

    //========================================
    //! \brief Active configuration while connected.
    //----------------------------------------
    TcpConfigurationPtr m_activeConfiguration;

}; // class TcpServer

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
