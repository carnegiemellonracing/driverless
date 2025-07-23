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
//! \date May 17, 2016
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/StatusCodes.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/commands/ecucommands/CommandEcuSetFilterC.hpp>
#include <microvision/common/logging/logging.hpp>

#include <boost/asio.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/optional/optional.hpp>

#include <set>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class for accepting connections via TCP/IP and sending data
//!        via this connection.
//! \date May 17, 2016
//------------------------------------------------------------------------------
class IdcTcpIpAcceptorBase
{
public:
    //========================================
    //! \brief Convenience ptr for boost shared_ptr.
    //----------------------------------------
    using Ptr = std::shared_ptr<IdcTcpIpAcceptorBase>;

    //========================================
    //! \brief Class which holds the current connection.
    //! \date May 17, 2016
    //----------------------------------------
    class SessionBase
    {
    public:
        //========================================
        //! \brief Convenience ptr for shared_ptr.
        //----------------------------------------
        using Ptr = std::shared_ptr<SessionBase>;

    public:
        //========================================
        //! \brief Creates a SessionBase.
        //!
        //! \param[in] io_service      Service which
        //!                            established connection.
        //! \param[in] deviceId        Device id of our simulated device
        //!                            needed for idc data header.
        //----------------------------------------
        SessionBase(IdcTcpIpAcceptorBase* const parent,
                    boost::asio::io_service& io_service,
                    const uint8_t deviceId = 1);

        //========================================
        //! \brief Creates a SessionBase.
        //!
        //! \param[in] io_service      Service which
        //! \param[in] rangeStart      Start of initial filter
        //!                            range which decides
        //!                            which DataContainer will be
        //!                            send via this session.
        //! \param[in] rangeEnd        End of initial filter
        //!                            range which decides
        //!                            which DataContainer will be
        //!                            send via this session.
        //!                            established connection.
        //! \param[in] deviceId        Device id of our simulated device
        //!                            needed for idc data header.
        //----------------------------------------
        SessionBase(IdcTcpIpAcceptorBase* const parent,
                    boost::asio::io_service& io_service,
                    const CommandEcuSetFilterC::RangeVector ranges,
                    const uint8_t deviceId = 1);

        //========================================
        //! \brief Destructs a SessionBase.
        //!
        //! Closes socket.
        //----------------------------------------
        virtual ~SessionBase();

        //========================================
        //! \brief Send a DataContainer.
        //! \param[in] dataContainer  the DataContainer which should be sent.
        //! \return The result whether the operation could be started or not.
        //! \sa ErrorCode
        //----------------------------------------
        StatusCode sendDataContainer(const microvision::common::sdk::SpecializedDataContainer& dataContainer,
                                     const ExporterBase& exporter);
        StatusCode sendStreamData(const IdcDataHeader& dh, const char* const dataPayloadBuffer);

        //========================================
        //! \brief Starts listening thread.
        //----------------------------------------
        void startListening() { startListen(); }

        //========================================
        //! \brief Checks whether connection is valid.
        //! \return \c True if connection is still valid,
        //!         \c false otherwise.
        //----------------------------------------
        bool isConnectionValid() { return (m_connectionALive && m_socket.is_open()); }

    public:
        boost::asio::ip::tcp::socket& getSocket() { return m_socket; }
        bool getConnectionALive() { return m_connectionALive; }

    public:
        void setConnectionALive(const bool connectionALive) { m_connectionALive = connectionALive; }

        //========================================
        //!\brief Set the ranges for filtering.
        //!\param[in] rangeVector  RangeVector to be
        //!                        applied.
        //----------------------------------------
        void setRanges(const CommandEcuSetFilterC::RangeVector& rangeVector) { m_ranges = rangeVector; }

        void setSizeOfPrevMsg(const uint32_t sizePrevMsg) { m_sizePrevMsg = sizePrevMsg; }

    public:
        void cancelAsyncIos() { m_socket.cancel(); }

    protected:
        //========================================
        //! \brief Worker function for m_ListenThreadPtr
        //!        needs to be overloaded by child.
        //----------------------------------------
        virtual void startListen() = 0;

    protected:
        IdcTcpIpAcceptorBase* const m_parent;

        //========================================
        //! \brief Socket which holds the connection.
        //----------------------------------------
        boost::asio::ip::tcp::socket m_socket;

        //========================================
        //! \brief Holds device id, needed when
        //!        writing an IdcDataHeader.
        //! \note Default value is 1.
        //----------------------------------------
        uint8_t m_deviceId;

    protected:
        //========================================
        //! \brief Lock sending function
        //!
        //! Can be called by listening thread, when replying,
        //! but also by other writing threads.
        //----------------------------------------
        boost::mutex m_sendMutex;

        //========================================
        //! \brief Saves the size of the last send msg.
        //!
        //! Needed when writing an IdcDataHeader when
        //! sending a DataContainer or replying to commands.
        //----------------------------------------
        uint32_t m_sizePrevMsg;

        //========================================
        //! \brief Flag which holds information/guess
        //!        whether the socket is still connected.
        //----------------------------------------
        bool m_connectionALive;

        //========================================
        //!\brief Ranges of datatypes to be be sent.
        //----------------------------------------
        CommandEcuSetFilterC::RangeVector m_ranges;

        std::vector<char> m_sendBuffer;
    }; //IdcTcpIpAcceptorBase::SessionBase

public:
    //========================================
    //! \brief Creates an IdcTcpIpAcceptorBase.
    //!
    //! \param[in] port  Port number for the connection.
    //----------------------------------------
    IdcTcpIpAcceptorBase(const unsigned short port = 12002);

    //========================================
    //! \brief Creates an IdcTcpIpAcceptorBase.
    //!
    //! \param[in] port            Port number for the connection.
    //----------------------------------------
    IdcTcpIpAcceptorBase(const boost::asio::deadline_timer::duration_type writeExpirationTime,
                         const unsigned short port = 12002);

    //========================================
    //! \brief Creates an IdcTcpIpAcceptorBase.
    //!
    //! \param[in] port            Port number for the connection.
    //----------------------------------------
    IdcTcpIpAcceptorBase(const boost::optional<boost::asio::deadline_timer::duration_type> writeExpirationTime,
                         const unsigned short port = 12002);

    //========================================
    //! \brief Destructor of the IdcTcpIpAcceptor class.
    //!
    //! Stopping io service thread and
    //! destroying the socket.
    //----------------------------------------
    virtual ~IdcTcpIpAcceptorBase();

public:
    //========================================
    //! \brief Initialization for acceptor.
    //!        Start to accept new connections.
    //----------------------------------------
    void init();

    //========================================
    //! \brief Returns whether there is at least
    //!        one session active.
    //! \return \c True if at least one session is active.
    //!         \c false otherwise.
    //----------------------------------------
    bool hasSessions() const { return !m_sessions.empty(); }

    //========================================
    //! \brief Get the number of active sessions.
    //! \return The number of activate sessions.
    //----------------------------------------
    int getNbOfSession() const { return int(m_sessions.size()); }

    //========================================
    //! \brief Sends \a dataContainer to all open connections.
    //!        And wait till the writes have finished.
    //! \param[in] dataContainer  The DataContainer which should
    //!                       be sent.
    //! \return The result of the operation.
    //! \sa StatusCode
    //----------------------------------------
    StatusCode sendDataContainer(const SpecializedDataContainer& dataContainer, const ExporterBase& exporter);
    StatusCode sendRawData(const IdcDataHeader& dh, const char* const dataPayloadBuffer);

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::IdcTcpIpAcceptorBase";
    static microvision::common::logging::LoggerSPtr logger;

protected:
    //========================================
    //! \brief Sends \a dataContainer to all open connections.
    //! \param[in] dataContainer  The DataContainer which should
    //!                       be sent.
    //----------------------------------------
    void issueWriteOperations(const SpecializedDataContainer& dataContainer, const ExporterBase& exporter);
    void issueRawWriteOperations(const IdcDataHeader& dh, const char* const dataPayloadBuffer);

    //========================================
    //! \brief Wait for all write operations to
    //!        be completed.
    //----------------------------------------
    void waitForWriteOperationsBeCompleted();

    //========================================
    //! \brief Handler for timeout situations.
    //! \param[in] session              Session that started
    //!                                 the deadline timer.
    //! \param[in] error                Result of waiting for
    //!                                 the deadline timer of
    //!                                 the \a session.
    //! \param[in] nbOfBytesTransfered  Number of bytes written
    //!                                 from the buffer. If an
    //!                                 error occurred this will be
    //!                                 less than the expected
    //!                                 size.
    //----------------------------------------
    void writeDone(SessionBase* const session,
                   const boost::system::error_code& error,
                   const std::size_t nbOfBytesTransfered);

    void writeTimeout(const boost::system::error_code& error);

    void cancelWriteOperations();

protected:
    //========================================
    //! \brief Gets current session ptr.
    //!
    //! Needs to be implemented by child classes.
    //! \return Session ptr casted to sessionBase.
    //----------------------------------------
    virtual SessionBase::Ptr getSessionPtr() = 0;

    //========================================
    //! \brief Gets new session ptr initialized with io_service.
    //! \param[in] io_service  Service which handles connections.
    //! \return Session ptr casted to sessionBase.
    //----------------------------------------
    virtual SessionBase::Ptr getNewSessionPtr(boost::asio::io_service& io_service) = 0;

private:
    //========================================
    //! \brief Working function for accepting new
    //!        requests from m_acceptIOServiceThreadPtr.
    //----------------------------------------
    void acceptorIOServiceThread();

    //========================================
    //! \brief Handles accepts, running in context
    //!        of m_acceptIOServiceThreadPtr.
    //----------------------------------------
    void handleAccept(const boost::system::error_code& error);

    //========================================
    //! \brief Closing acceptor.
    //----------------------------------------
    void closeAcceptor();

private:
    //========================================
    //! \brief Session ptr which will handle the
    //!        next accept.
    //!
    //! Will be append to m_sessions when a new
    //! session has been accepted. And a new
    //! session pointer will be created for the
    //! next session to be accepted.
    //----------------------------------------
    SessionBase::Ptr m_sessionPtr;

    //========================================
    //! \brief Service which handles accepts and
    //!        then handed over to sessions.
    //----------------------------------------
    boost::asio::io_service m_ioService;

private:
    //========================================
    //! \brief Prevent sessions to be accessed
    //!       simultaneously by multiple threads.
    //----------------------------------------
    boost::mutex m_sessionsMutex;

    //========================================
    //! \brief Vector which holds all open connections.
    //!
    //! If connection is detected as being closed
    //! (e.g during writing) it will be deleted.
    //----------------------------------------
    std::vector<SessionBase::Ptr> m_sessions;

private: // used by write operation
    //========================================
    //! \brief Maps holds for each session the
    //!        expected number of bytes to be
    //!        send.
    //----------------------------------------
    std::map<SessionBase*, uint32_t> m_activeSending;

    //========================================
    //! \brief Current status of the write operations
    //!        to all sessions.
    //----------------------------------------
    StatusCode m_sendingStatus;

    //========================================
    //! \brief Mutex to guard \a m_activeSending.
    //----------------------------------------
    boost::mutex m_writeCondMutex;

    //========================================
    //! \brief Condition to signaling whether
    //!        all write operations have been
    //!        completed.
    //----------------------------------------
    boost::condition_variable writeCondition;

    //========================================
    //! \brief Expiring duration for write operations.
    //----------------------------------------
    boost::asio::deadline_timer::duration_type m_writeExpirationPeriod;

    //========================================
    //! \brief Deadline timer for write operations.
    //----------------------------------------
    boost::asio::deadline_timer m_writeExpirationTimer;

    enum class WriteState : uint8_t
    {
        Idle,
        InProgress,
        Error,
        TimeOut,
        Completed
    };

    WriteState m_writeState;

private:
    //========================================
    //! \brief Accepting requests.
    //----------------------------------------
    boost::asio::ip::tcp::acceptor m_acceptor;

    //========================================
    //! \brief Thread waiting for connection requests.
    //----------------------------------------
    boost::scoped_ptr<boost::thread> m_acceptIOServiceThreadPtr;
}; //IdcTcpIpAcceptorBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
