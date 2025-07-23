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
//! \date Aug 21, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/ThreadRunner.hpp>
#include <microvision/common/logging/logging.hpp>

#include <microvision/common/sdk/misc/defines/boost.hpp>

#include <boost/asio.hpp>
#include <boost/thread/mutex.hpp>

#include <functional>
#include <map>

//==============================================================================

namespace basio = boost::asio;

//==============================================================================F
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class serves as a de-multiplexer for packets received via local UDP endpoints. The packets are
//! separated according to their remote IP address and/or port number. So, clients using this class can register
//! a callback for UDP traffic flowing from a remote to a local endpoint.
//! In general, it is not allow to have more than one thread listening on the same UDP socket. So, instances of this
//! class have an own thread that receives UDP packets and distributes them via the registered callbacks.
//------------------------------------------------------------------------------
class MICROVISION_SDK_API UdpDemultiplexer final
{
public:
    using UdpDemultiplexerPtr     = std::shared_ptr<UdpDemultiplexer>;
    using UdpDemultiplexerWeakPtr = std::weak_ptr<UdpDemultiplexer>;

    //! Handle to identify callback registrations.
    using CallbackHandle = const void*;

    //! Type of function to be called when data was received.
    using DataBufferPtr    = std::shared_ptr<std::vector<char>>;
    using CallbackFunction = std::function<void(const CallbackHandle, const DataBufferPtr&)>;

public: // constructor/destructor
    //========================================
    //! \brief Get an instance of this class that listens on a specific port on all local network interfaces.
    //!
    //! \param[in] localPort      Port number of the local network interface to listen on. The port number must not be
    //!                           zero!
    //! \param[in] maxBufferSize  Maximum size of a UDP packet sent to this port.
    //! \return                   If an instance corresponding to the parameters could be found or created, this
    //!                           instance is returned. Otherwise this function returns \c nullptr.
    //!
    //! This functions fails (i.e. returns \c nullptr) if:
    //! <ul>
    //! <li>an instance corresponding to the given parameters has been created before and there are already callbacks
    //! registered at this instance (i.e. the reception of UDP packets is already running) and the given maximum packet
    //! size exceeds the one of the previously created instance</li>
    //! </ul>
    //----------------------------------------
    static UdpDemultiplexerPtr get(const uint16_t localPort, const std::size_t maxBufferSize)
    {
        return get(basio::ip::udp::endpoint(basio::ip::address_v4::any(), localPort), maxBufferSize);
    }

    //========================================
    //! \brief Get an instance of this class that listens on a specific port on a specific network interface.
    //!
    //! \param[in] localAddress   IPv4 address of the local network interface to listen on.
    //! \param[in] localPort      Port number of the local network interface to listen on. The port number must not be
    //!                           zero!
    //! \param[in] maxBufferSize  Maximum size of a UDP packet sent to this port.
    //! \return                   If an instance corresponding to the parameters could be found or created, this
    //!                           instance is returned. Otherwise this function returns \c nullptr.
    //!
    //! If the IP address is set to 0.0.0.0 (any address) the instance listens on all local network interfaces.<br>
    //! This functions fails (i.e. returns \c nullptr) if:
    //! <ul>
    //! <li>an instance corresponding to the given parameters has been created before and there are already callbacks
    //! registered at this instance (i.e. the reception of UDP packets is already running) and the given maximum packet
    //! size exceeds the one of the previously created instance</li>
    //! <li>the given IP address is set to listen on all local network interfaces and an instance has been created for
    //! the given port before that is listening on a single network interface only</li>
    //! <li>the given IP address is set to listen on a single local network interface only and an instance has been
    //! created for the given port before that is listening on all network interfaces</li>
    //! </ul>
    //----------------------------------------
    static UdpDemultiplexerPtr
    get(const basio::ip::address_v4& localAddress, const uint16_t localPort, const std::size_t maxBufferSize)
    {
        return get(basio::ip::udp::endpoint(localAddress, localPort), maxBufferSize);
    }

    //========================================
    //! \brief Get an instance of this class that listens on a specific endpoint.
    //!
    //! \param[in] localEndpoint  IP address and port of the local network interface to listen on. The IP address part
    //!                           of this endpoint must be IPv4 and the port number must not be zero!
    //! \param[in] maxBufferSize  Maximum size of a UDP packet sent to this port.
    //! \return                   If an instance corresponding to the parameters could be found or created, this
    //!                           instance is returned. Otherwise this function returns \c nullptr.
    //!
    //! If the IP address part of the endpoint is set to 0.0.0.0 (any address) the instance listens on all local
    //! network interfaces.<br>
    //! This functions fails (i.e. returns \c nullptr) if:
    //! <ul>
    //! <li>an instance corresponding to the given parameters has been created before and there are already callbacks
    //! registered at this instance (i.e. the reception of UDP packets is already running) and the given maximum packet
    //! size exceeds the one of the previously created instance</li>
    //! <li>the given IP address is set to listen on all local network interfaces and an instance has been created for
    //! the given port before that is listening on a single network interface only</li>
    //! <li>the given IP address is set to listen on a single local network interface only and an instance has been
    //! created for the given port before that is listening on all network interfaces</li>
    //! </ul>
    //----------------------------------------
    static UdpDemultiplexerPtr get(const basio::ip::udp::endpoint& localEndpoint, const std::size_t maxBufferSize);

    //========================================
    //!\brief Destructor.
    //----------------------------------------
    virtual ~UdpDemultiplexer();

public:
    //========================================
    //! \brief Register a callback function to be called when UDP packets from the specified remote port were
    //! received.
    //!
    //! \param[in] remotePort  Port number of the remote endpoint to receive from.
    //! \param[in] callback    The function to be called when UDP packets from the specified remote endpoint were
    //!                        received.
    //! \return                Handle used to identify the registration (e.g. when un-registering the callback).
    //!
    //! If the port number is set to zero the callback function is called for all received UDP packets.
    //----------------------------------------
    CallbackHandle registerCallback(const uint16_t remotePort, CallbackFunction callback)
    {
        return registerCallback(basio::ip::udp::endpoint(basio::ip::address_v4::any(), remotePort), callback);
    }

    //========================================
    //! \brief Register a callback function to be called when UDP packets from the specified remote IP address were
    //! received.
    //!
    //! \param[in] remoteAddress  IPv4 address of the remote endpoint to receive from.
    //! \param[in] callback       The function to be called when UDP packets from the specified remote endpoint were
    //!                           received.
    //! \return                   Handle used to identify the registration (e.g. when un-registering the callback).
    //!
    //! If the IP address part is set to 0.0.0.0 (any address) the callback function is called for all received UDP
    //! packets.
    //----------------------------------------
    CallbackHandle registerCallback(const basio::ip::address_v4& remoteAddress, CallbackFunction callback)
    {
        return registerCallback(basio::ip::udp::endpoint(remoteAddress, 0), callback);
    }

    //========================================
    //! \brief Register a callback function to be called when UDP packets from the specified remote endpoint were
    //! received.
    //!
    //! \param[in] remoteAddress  IPv4 address of the remote endpoint to receive from.
    //! \param[in] remotePort     Port number of the remote endpoint to receive from.
    //! \param[in] callback       The function to be called when UDP packets from the specified remote endpoint were
    //!                           received.
    //! \return                   Handle used to identify the registration (e.g. when un-registering the callback).
    //!
    //! If the port number is set to zero the callback function is called for all UDP packets sent from the given
    //! remote IP address. If the IP address part is set to 0.0.0.0 (any address) the function is called for all UDP
    //! packets received from any remote endpoint that used the remote port for sending. If both parts are zero, the
    //! callback function is called for all received UDP packets.
    //----------------------------------------
    CallbackHandle
    registerCallback(const basio::ip::address_v4& remoteAddress, const uint16_t remotePort, CallbackFunction callback)
    {
        return registerCallback(basio::ip::udp::endpoint(remoteAddress, remotePort), callback);
    }

    //========================================
    //! \brief Register a callback function to be called when UDP packets from the specified remote endpoint were
    //! received.
    //!
    //! \param[in] remoteEndpoint  IPv4 address and port number of the remote endpoint to receive from.
    //! \param[in] callback        The function to be called when UDP packets from the specified remote endpoint were
    //!                            received.
    //! \return                    Handle used to identify the registration (e.g. when un-registering the callback).
    //!
    //! If the port number part of the given endpoint is set to zero the callback function is called for all UDP
    //! packets sent from the given remote IP address. If the IP address part is set to 0.0.0.0 (any address)
    //! the function is called for all UDP packets received from any remote endpoint that used the remote port for
    //! sending. If both parts are zero, the callback function is called for all received UDP packets.
    //----------------------------------------
    CallbackHandle registerCallback(const basio::ip::udp::endpoint remoteEndpoint, CallbackFunction callback);

    //========================================
    //! \brief Un-register a previously registered callback function.
    //!
    //! \param[in] handle  Handle to identify the registration as returned by one of the register functions.
    //! \return            \c true if the handle was found and the registration was removed, \c false otherwise.
    bool unregisterCallback(const CallbackHandle handle);

    //========================================
    //! \brief Get the number of worker threads used for receiving UDP packets.
    //!
    //! \return            The number of worker threads used for receiving UDP packets.
    //----------------------------------------
    int getWorkerThreadCount() const { return m_workerThreadCount; }

    //========================================
    //! \brief Set the number of worker threads used for receiving UDP packets.
    //!
    //! \param[in] value  New number of worker threads used for receiving UDP packets.
    //!
    //! The number of worker threads cannot be set while clients are registered.
    //----------------------------------------
    void setWorkerThreadCount(const int value);

protected:
    using UdpSocketPtr = std::shared_ptr<boost::asio::ip::udp::socket>;
    using PortMap      = std::map<basio::ip::udp::endpoint, UdpDemultiplexerWeakPtr>;

    struct CallbackEntry
    {
        basio::ip::udp::endpoint endpoint;
        CallbackFunction function;
    };
    using Callbacks = std::map<CallbackHandle, CallbackEntry>;

protected:
    static void cleanPortMap();
    static UdpDemultiplexerPtr findInstance(const boost::asio::ip::udp::endpoint& endpoint);

    // Do not use constructor, but factory methods to create instances!
    UdpDemultiplexer(const basio::ip::udp::endpoint& localEndpoint, const std::size_t maxBufferSize);

    void adjustBufferSize(const std::size_t maxBufferSize);
    virtual bool startThread(); // Shall be overridden for unittests only.
    virtual void stopThread(); // Shall be overridden for unittests only.
    virtual void udpReceiveThreadMain(); // Shall be overridden for unittests only.
    virtual void addIoServiceWork(); // Shall be overridden for unittests only.
    void handleUdpData(const std::shared_ptr<boost::asio::ip::udp::endpoint> remoteEndpoint,
                       const DataBufferPtr& dataBuffer,
                       const boost::system::error_code& errorCode,
                       size_t recvBytes);
    bool endpointMatches(const basio::ip::udp::endpoint& remoteEndpoint,
                         const basio::ip::udp::endpoint& callbackEndpoint);
    CallbackHandle getNextCallbackHandle();

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::UdpDemultiplexer";
    static microvision::common::logging::LoggerSPtr logger;
    static boost::mutex portMapMutex; // Guards the following members.
    static PortMap portMap;

protected:
    mutable boost::mutex m_callbacksMutex; // Guards the following members.
    Callbacks m_callbacks;
    ThreadRunner m_udpReceiveThread;

    uint32_t m_lastCallbackHandle{0};
    int m_workerThreadCount{2};
    basio::ip::udp::endpoint m_localEndpoint;
    BoostIoContext m_ioService;
    UdpSocketPtr m_socket{nullptr};
    std::size_t m_dataBufferSize;
    uint64_t m_nbOfRecvPackets{0};
}; // UdpDemultiplexer

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
