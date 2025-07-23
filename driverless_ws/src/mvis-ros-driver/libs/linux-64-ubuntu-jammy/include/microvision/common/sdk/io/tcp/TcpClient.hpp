//==============================================================================
//! \file
//!
//! \brief TCP client network interface.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 28, 2022
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
//! \brief TCP client network interface.
//! \extends TcpBase
//! \extends DataPackageNetworkSender
//! \extends DataPackageNetworkReceiver
//------------------------------------------------------------------------------
class TcpClient final : public TcpBase, public DataPackageNetworkSender, public DataPackageNetworkReceiver
{
public:
    //========================================
    //! \brief Number of threads instantiated for asynchronous network communication.
    //----------------------------------------
    static constexpr uint8_t defaultNumberOfThreads{2};

    //========================================
    //! \brief Get type name of this network interface.
    //! \returns NetworkInterface type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getTypeName();

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::TcpClient";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

    //========================================
    //! \brief Implementation of TCP io communication.
    //! \extends TcpBase::ConnectionSupport
    //! \extends TcpBase::TimeoutSupport
    //----------------------------------------
    class ClientImpl : public ConnectionSupport, public TimeoutSupport
    {
    public:
        //========================================
        //! \brief Constructor with parameters for io communication.
        //! \param[in] context                 Boost io context.
        //! \param[in] numberOfNetworkThreads  Number of threads for asynchronous network communication.
        //----------------------------------------
        ClientImpl(BoostIoContext& context, uint8_t numberOfNetworkThreads);

        //========================================
        //! \brief Default destructor.
        //----------------------------------------
        ~ClientImpl();

    public:
        //========================================
        //! \brief Start up all threads for asynchronous network communication.
        //----------------------------------------
        void start();

        //========================================
        //! \brief Shutdown all threads for asynchronously network communication.
        //----------------------------------------
        void stop();

        //========================================
        //! \brief Run io context to use this thread for receiving/sending data etc.
        //! \return Either \c true if socket is still open, otherwise \c false.
        //----------------------------------------
        bool run();

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

    public: // implements ConnectionSupport
        //========================================
        //! \brief Will be called when a data package is received.
        //! \param[in] package  Received data package.
        //----------------------------------------
        void onDataReceived(const NetworkDataPackagePtr& package) override;

        //========================================
        //! \brief Will be called when a connection is disconnected.
        //! Disconnection may be unintended. Depending on reconnect mode (\sa m_connectionMode) the client will try to reconnect.
        //! \param[in] inIdle  Either \c true if data queue is empty, otherwise \c false.
        //----------------------------------------
        void onDataSent(const bool inIdle) override;

        //========================================
        //! \brief Will be called when a connection is disconnected.
        //----------------------------------------
        void onDisconnect() override;

        //========================================
        //! \brief Check wether the connection is established.
        //! \return Either \c true if connection is established, otherwise \c false.
        //----------------------------------------
        bool isConnected() const override;

    public: // implements TimeoutSupport
        //========================================
        //! \brief Will be called when the timeout occurred.
        //----------------------------------------
        void onTimeoutOccurred() override;

        //========================================
        //! \brief Will be called when the timer is canceled.
        //----------------------------------------
        void onCancel() override;

    private:
        //========================================
        //! \brief Connect without blocking against TCP server.
        //----------------------------------------
        void asyncConnect();

        //========================================
        //! \brief TCP connection handler.
        //! Called after connection is established or not successful.
        //! \param[in] error  Boost error.
        //----------------------------------------
        void connectHandler(const boost::system::error_code& error);

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
        //! \brief Remote endpoint of TCP server.
        //----------------------------------------
        TcpBase::BoostEndpointType m_remoteEndpoint;

        //========================================
        //! \brief Either \c true if client is connected, otherwise \c false.
        //----------------------------------------
        std::atomic_bool m_isConnected;

        //========================================
        //! \brief Connection timeout after milliseconds.
        //----------------------------------------
        uint32_t m_connectionTimeout;

        //========================================
        //! \brief Communication timeout after milliseconds.
        //----------------------------------------
        uint32_t m_communicationTimeout;

        //========================================
        //! \brief Reconnection mode.
        //----------------------------------------
        TcpConfiguration::ReconnectMode m_connectionMode;

        //========================================
        //! \brief Output queue observer thread.
        //----------------------------------------
        BackgroundWorker m_outputQueueObserver;

        //========================================
        //! \brief Thread resources sync handler.
        //----------------------------------------
        ThreadSyncPtr m_syncHandle;

        //========================================
        //! \brief All threads used for asynchronous network communication.
        //----------------------------------------
        std::vector<BackgroundWorkerUPtr> m_clientThreads;

        //========================================
        //! \brief Data output queue for sending data.
        //----------------------------------------
        ThreadSafe<DataPackageNetworkSender::QueueType> m_outputQueue;

        //========================================
        //! \brief Data input queue for receiving data.
        //----------------------------------------
        ThreadSafe<DataPackageNetworkReceiver::QueueType> m_inputQueue;

        //========================================
        //! \brief Either \c true if session will disconnected externally, otherwise \c false.
        //----------------------------------------
        std::atomic_bool m_externalDisconnect;

    }; // class ClientImpl

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    TcpClient();

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    TcpClient(TcpClient&& other) = delete;

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    TcpClient(const TcpClient& other) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~TcpClient() override;

public:
    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    TcpClient& operator=(TcpClient&& other) = delete;

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    TcpClient& operator=(const TcpClient& other) = delete;

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
    //! \brief Checks if a connection is established.
    //! \returns Either \c true if connection is established or otherwise \c false.
    //----------------------------------------
    bool isConnected() const override;

    //========================================
    //! \brief Checks if a connection is established and packages are still processed.
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
    //! \brief Get ThreadSafe::Access to the output queue where all data ready to be sent will be stored.
    //! \returns ThreadSafe::Access to the output queue.
    //----------------------------------------
    ThreadSafe<DataPackageNetworkSender::QueueType>::Access getOutputQueue() override;

public: // implements DataPackageNetworkReceiver
    //========================================
    //! \brief Get ThreadSafe::Access to the input queue where all received data will be stored.
    //! \returns ThreadSafe::Access to the input queue.
    //----------------------------------------
    ThreadSafe<DataPackageNetworkReceiver::QueueType>::Access getInputQueue() override;

private:
    //========================================
    //! \brief Boost io context.
    //----------------------------------------
    BoostIoContext m_context;

    //========================================
    //! \brief TCP client io communication.
    //----------------------------------------
    ClientImpl m_client;

    //========================================
    //! \brief Active configuration while connected.
    //----------------------------------------
    TcpConfigurationPtr m_activeConfiguration;

}; // class TcpClient

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================