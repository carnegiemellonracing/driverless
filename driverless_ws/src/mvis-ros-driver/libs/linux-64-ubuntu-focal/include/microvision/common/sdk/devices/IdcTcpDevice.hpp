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
//! \date Jul 2, 2018
//!
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcEthDevice.hpp>
#include <microvision/common/sdk/devices/TcpReceiveThreadEnsemble.hpp>

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
//!\brief IdcDevice for use with TCP
//!\date Jul 2, 2018
//!
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED IdcTcpDevice : public IdcEthDevice
{
public:
    using ReconnectMode = TcpReceiveThreadEnsemble::ReconnectMode;
    using TimeDuration  = boost::posix_time::time_duration;

public:
    //========================================
    //!\brief Create an IdcTcpDevice.
    //!
    //! This constructor will create an IdcDevice class object
    //! given by the template class DeviceImpl.
    //!
    //! \param[in] ip      IP address of the device
    //!                    to be connected with.
    //! \param[in] port    Port number for the device.
    //----------------------------------------
    IdcTcpDevice(const std::string& ip, const uint16_t port) : IdcEthDevice(), m_strIP(ip), m_port(port) {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~IdcTcpDevice() override = default;

    //========================================
    //! \brief Get the reconnect mode for this device.
    //!
    //! \return The reconnect mode, i.e. if and how the device will automatically reconnect to the hardware.
    //----------------------------------------
    ReconnectMode getReconnectMode() const { return m_reconnectMode; }

    //========================================
    //! \brief Set the reconnect mode for this device.
    //!
    //! \param[in] reconnectMode  The new reconnect mode.
    //----------------------------------------
    void setReconnectMode(const ReconnectMode reconnectMode) { m_reconnectMode = reconnectMode; }

    //========================================
    //! \brief Get the time between two attempts to reconnect.
    //!
    //! \return The reconnect time.
    //----------------------------------------
    const TimeDuration& getReconnectTime() const { return m_reconnectTime; }

    //========================================
    //! \brief Set the time between two attempts to reconnect.
    //!
    //! \param[in] reconnectTime  The new reconnect time.
    //----------------------------------------
    void setReconnectTime(const TimeDuration& reconnectTime) { m_reconnectTime = reconnectTime; }

public:
    virtual void connect(const uint32_t timeoutSec = IdcEthDevice::defaultReceiveTimeoutSeconds) override
    {
        boost::system::error_code ec;
        boost::asio::ip::address ipAdr = boost::asio::ip::address::from_string(m_strIP, ec);

        if (ec)
        {
            LOGERROR(this->m_logger, "Invalid IP address (" << m_strIP << ") " << ec.message());
            IdcDeviceBase::Lock lock(this->m_mutex);
            this->m_threadState = IdcDeviceBase::ThreadState::StartFailed;
            return;
        }

        if (this->m_receiveThread)
        {
            LOGWARNING(this->m_logger, "Receive thread already running.");
            return;
        }

        TcpReceiveThreadEnsemble::ReceiveDataFunc rd
            = boost::bind(&IdcEthDevice::receivedDataContainer, this, boost::placeholders::_1, boost::placeholders::_2);
        TcpReceiveThreadEnsemble* receiveThread = new TcpReceiveThreadEnsemble(ipAdr, m_port, rd);
        receiveThread->setReconnectMode(m_reconnectMode);
        receiveThread->setReconnectTime(m_reconnectTime);
        this->m_receiveThread = receiveThread;

        IdcEthDevice::connect(timeoutSec);
    }

protected:
    std::string m_strIP;
    unsigned short m_port{0};
    ReconnectMode m_reconnectMode{ReconnectMode::WaitForRemoteDevice};
    TimeDuration m_reconnectTime{boost::posix_time::millisec(100)};
}; // IdcDevice

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
