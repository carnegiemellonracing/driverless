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
//! \date July 20, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcDeviceBase.hpp>
#include <microvision/common/sdk/devices/ThreadEnsemble.hpp>

#include <boost/asio.hpp>
#include <boost/optional.hpp>

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
//!\class IdcEthDevice
//!\brief Base class for all idc devices connected via TCP/IP or UDP.
//!\date July 20, 2017
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED IdcEthDevice : public IdcDeviceBase
{
public:
    using UdpSocketPtr = std::shared_ptr<boost::asio::ip::udp::socket>;

public:
    //========================================
    //!\brief Create an IdcEthDevice (general device class).
    //!
    //! This constructor will create an IdcEthDevice class object
    //! for devices communicating by network.
    //----------------------------------------
    IdcEthDevice();
    virtual ~IdcEthDevice();
    //========================================

private:
    IdcEthDevice(const IdcEthDevice&); // forbidden
    IdcEthDevice& operator=(const IdcEthDevice&); // forbidden

public:
    void receivedDataContainer(const IdcDataHeader* dh, ConstCharVectorPtr bodyBuf);

    //========================================
    //!\brief Called whenever a serialized data container is fully received from input stream.
    //!
    //! The full idc data block is imported with the given importer.
    //!
    //! \param[in] dh           Received idc data header for the received data block.
    //! \param[in, out] is      The input stream from which the data is read.
    //! \param[in] containerId  The identification of the data container.
    //! \param[in] importer     An importer to deserialize the data block received.
    //! \return The deserialized data container.
    //----------------------------------------
    const std::shared_ptr<DataContainerBase> onDataReceived(const IdcDataHeader& dh,
                                                            std::istream& is,
                                                            const DataContainerBase::IdentificationKey& containerId,
                                                            ImporterBase*& importer) override;

    //========================================
    //!\brief Establish the connection to the
    //!       hardware.
    //!
    //! Starting the receiving thread.
    //!\param[in] timeoutSec  Device timeout in seconds
    //----------------------------------------
    void connect(const uint32_t timeoutSec = defaultReceiveTimeoutSeconds) override;

    //========================================
    //!\brief Disconnect the TCP/IP connection
    //!       to the hardware device.
    //----------------------------------------
    void disconnect() override;

    //========================================
    //!\brief Checks whether the TCP/IP connection to the hardware device is established and can receive data.
    //!
    //!\return \c True, if messages from the hardware can be received, \c false otherwise.
    //----------------------------------------
    bool isConnected() const override;

    //========================================
    //!\brief Checks whether the thread for handling TCP/IP connections to the hardware is running.
    //!
    //!\return \c True, if the thread is running, \c false otherwise.
    //!
    //!\note This should not be mixed up with \a isConnected(). A device is running if the corresponding thread is
    //!      running, no matter if the connection to the hardware is established or not.
    //----------------------------------------
    bool isRunning() const override;

    //========================================
    //! \brief Get the timeout period of the receive operation.
    //! \return  Number of seconds.
    //----------------------------------------
    uint32_t getRecvTimeoutSeconds() const;

    //========================================
    //! \brief Get the timeout period of the receive operation.
    //! \return  Number of milliseconds.
    //----------------------------------------
    uint32_t getReceiveTimeoutInMs() const;

    //========================================
    //!\brief Set the timeout period of the receive
    //!       operation.
    //!\note: if a connection to a hardware device is already established, the
    //!       new timeout takes effect immediately.
    //!       If no device is connected, changes take effect after connecting
    //!       to a device using the method getConnected().
    //! \param[in] seconds  Number of seconds.
    //----------------------------------------
    void setRecvTimeoutSeconds(const uint32_t seconds);

    //========================================
    //! \brief Set the timeout period of the receive operation.
    //! \note: if a connection to a hardware device is already established, the
    //!        new timeout takes effect immediately.
    //!        If no device is connected, changes take effect after connecting
    //!        to a device using the method getConnected().
    //!  \param[in] milliseconds  Number of milliseconds.
    //----------------------------------------
    void setReceiveTimeoutInMs(const uint32_t miliseconds);

    //========================================
    //!\brief Send a command which expects no reply.
    //!\param[in] cmd  Command to be sent.
    //!\return The result of the operation.
    //!\sa ErrorCode
    //----------------------------------------
    StatusCode sendCommand(const CommandCBase& cmd, const SpecialExporterBase<CommandCBase>& exporter) final;

    //========================================
    //!\brief Send a command and wait for a reply.
    //!
    //! The command will be sent. The calling thread
    //! will sleep until a reply has been received
    //! but not longer than the number of milliseconds
    //! given in \a timeOut.
    //!
    //!\param[in]       cmd    Command to be sent.
    //!\param[in, out]  reply  The reply container for
    //!                        the reply to be stored into.
    //!\param[in]       timeOut  Number of milliseconds to
    //!                          wait for a reply.
    //!\return The result of the operation.
    //!\sa ErrorCode
    //----------------------------------------
    StatusCode sendCommand(const CommandCBase& cmd,
                           const SpecialExporterBase<CommandCBase>& exporter,
                           CommandReplyBase& reply,
                           const boost::posix_time::time_duration timeOut = boost::posix_time::milliseconds(500)) final;

protected:
    static constexpr uint32_t defaultReceiveTimeoutSeconds{10};

    Mutex m_mutex;
    ThreadState m_threadState{ThreadState::NotRunning};
    ThreadEnsemble* m_receiveThread{nullptr};

private:
    uint32_t m_receiveTimeoutInMs{}; // hardware timeout in milliseconds

}; // IdcEthDevice

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
