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
//! \date Jun 02 2016
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcTcpIpAcceptorBase.hpp>

#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class IdcTcpIpAcceptorScala
//! \brief Class for accepting connections via TCP/IP and sending data
//!        via this connection, behaving like a Scala.
//! \date Jun 02, 2016
//------------------------------------------------------------------------------
class IdcTcpIpAcceptorScala final : public IdcTcpIpAcceptorBase
{
public:
    //========================================
    //! \brief Convenience using std::shared_ptr<IdcTcpIpAcceptorScala> = for std::shared_ptr.
    //! using Ptr =;
    //----------------------------------------

private:
    //========================================
    //! \brief Handles connections and act like a Scala sensor.
    //! \date Jun 02, 2016
    //----------------------------------------
    class SessionScala final : public IdcTcpIpAcceptorBase::SessionBase
    {
    public:
        //========================================
        //! \brief Convenience using std::shared_ptr<SessionScala> = for std::shared_ptr.
        //! using Ptr =;
        //! ----------------------------------------

    public:
        //========================================
        //! \brief Creates a SessionScala
        //!
        //! \param[in] io_service      Service which
        //!                            established connection.
        //! \param[in] deviceId        Id of our simulated device
        //!                            needed for idc data header.
        //----------------------------------------
        SessionScala(IdcTcpIpAcceptorBase* const parent,
                     boost::asio::io_service& io_service,
                     const uint8_t deviceId = 1);

        //========================================
        //! \brief Destructs a SessionScala.
        //----------------------------------------
        virtual ~SessionScala() {}

    protected:
        //========================================
        //! \brief Worker function for m_ListenThreadPtr.
        //!
        //! Doing nothing. A Scala sensor does not
        //! listen to ethernet commands.
        //----------------------------------------
        virtual void startListen();
    }; //IdcTcpIpAcceptorScala::SessionScala

protected:
    static const int msgBufferSize = 4 * 65536;

public:
    //========================================
    //! \brief Creates an IdcTcpIpAcceptorScala.
    //!
    //! \param[in] port            Port number for the
    //!                            connection.
    //----------------------------------------
    IdcTcpIpAcceptorScala(const unsigned short port = 12004);

    //========================================
    //! \brief Creates an IdcTcpIpAcceptorScala.
    //!
    //! \param[in  writeExpirationTime  Timeout limit for async
    //!                                 write operations.
    //! \param[in] port                 Port number for the
    //!                                 connection.
    //----------------------------------------
    IdcTcpIpAcceptorScala(const boost::asio::deadline_timer::duration_type writeExpirationTime,
                          const unsigned short port = 12004);

    //========================================
    //! \brief Destructs an IdcTcpIpAcceptorScala.
    //----------------------------------------
    virtual ~IdcTcpIpAcceptorScala() {}

protected:
    //========================================
    //! \brief Gets current session ptr.
    //! \return Session ptr casted to sessionBase.
    //----------------------------------------
    virtual IdcTcpIpAcceptorBase::SessionBase::Ptr getSessionPtr() { return m_sessionPtr; }

    //========================================
    //! \brief Gets new session ptr initialized
    //!        with io_service.
    //! \param[in] io_service  Service which handles
    //!                        connections.
    //! \return Session ptr casted to sessionBase.
    //----------------------------------------
    virtual IdcTcpIpAcceptorBase::SessionBase::Ptr getNewSessionPtr(boost::asio::io_service& io_service);

private:
    //========================================
    //! \brief Session ptr for next session which
    //!        will be established.
    //----------------------------------------
    SessionScala::Ptr m_sessionPtr;
}; // IdcTcpIpAcceptorScala

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
