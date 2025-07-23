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
//! \date Apr 7, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/listener/DataContainerListener.hpp>
#include <microvision/common/sdk/listener/IdcDataPackageListener.hpp>
#include <microvision/common/sdk/datablocks/ImporterBase.hpp>
#include <microvision/common/sdk/datablocks/commands/Command2010.hpp>
#include <microvision/common/sdk/datablocks/commands/Command.hpp>
#include <microvision/common/sdk/CommandId.hpp>
#include <microvision/common/sdk/misc/StatusCodes.hpp>
#include <microvision/common/sdk/MsgBufferBase.hpp>
#include <microvision/common/sdk/EventMonitor.hpp>

#include <microvision/common/logging/logging.hpp>

#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/iostreams/stream.hpp>

#include <unordered_map>
#include <iostream>
#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class IdcDataHeader;
class DataContainerBase;
class DataStreamer;
class ImporterBase;

//==============================================================================
//! \brief Base class for all idc devices.
//!
//! A device has a connection to a source of data, maintains a state and can notify registered listeners for new data.
//! Also control commands can be received to change state.
//!
//! \note This is thread-safe. Listeners can be registered from different threads.
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED IdcDeviceBase : private boost::noncopyable
{
public:
    using ConstCharVectorPtr = MsgBufferBase::ConstCharVectorPtr;

protected:
    enum class ThreadState : uint8_t
    {
        NotRunning  = 0,
        Starting    = 1,
        Running     = 2,
        Stopping    = 3,
        StartFailed = 4,
        RunFailed   = 5
    }; // ThreadState

    //========================================
    //!\brief Type of mutex guard.
    //----------------------------------------
    using Lock = boost::mutex::scoped_lock;

    //========================================
    //!\brief Type of mutex.
    //----------------------------------------
    using Mutex = boost::mutex;

    //========================================
    //!\brief List of DataContainerListenerBase (pointer).
    //----------------------------------------
    using ContainerListenerList = std::list<DataContainerListenerBase*>;

    //========================================
    //!\brief ContainerListenerListMap maps a pair of DataTypeId/ContainerHash to a list of all
    //!       registered ContainerListener for this combination.
    //----------------------------------------
    using ContainerListenerListMap
        = std::unordered_map<DataContainerListenerBase::DataTypeHashUuid, ContainerListenerList>;

    //========================================
    //!\brief RegisteredContainerListenerListMap maps a DataTypeId to all
    //!       registered ContainerListener for this data type.
    //----------------------------------------
    using RegisteredContainerListenerListMap = std::unordered_map<DataTypeId, ContainerListenerListMap>;

    //========================================
    //!\brief List of DataStreamer (pointer).
    //----------------------------------------
    using StreamerList = std::list<DataStreamer*>;

    //========================================
    //!\brief List of IdcDataPackageListener (pointer).
    //----------------------------------------
    using IdcDataPackageListenerList = std::list<IdcDataPackageListener*>;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    IdcDeviceBase();

    //========================================
    //! \brief Destructor
    //----------------------------------------
    virtual ~IdcDeviceBase();

public:
    //========================================
    //!\brief Establish the connection to the hardware.
    //!
    //! Starting the receiving thread.
    //!
    //!\param[in] timeoutSec  Device timeout in seconds
    //----------------------------------------
    virtual void connect(const uint32_t timeoutSec) = 0;

    //========================================
    //!\brief Disconnect the connection to the hardware device.
    //----------------------------------------
    virtual void disconnect();

    //========================================
    //!\brief Checks whether the TCP/IP connection to the hardware device is established
    //!       and can receive data.
    //!\return \c True, if messages from the hardware can be received, \c false otherwise.
    //----------------------------------------
    virtual bool isConnected() const = 0;

    //========================================
    //!\brief Checks whether the thread for handling TCP/IP connections to the hardware is running.
    //!\return \c True, if the thread is running, \c false otherwise.
    //!\note This should not be mixed up with \a isConnected(). A device is running if the corresponding thread is
    //!      running, no matter if the connection to the hardware is established or not.
    //----------------------------------------
    virtual bool isRunning() const = 0;

    //========================================
    //!\brief Register a DataListener to this device.
    //!
    //! Each time a message has been received by
    //! this object, the registered listener will be
    //! called which are listening to the received message
    //! type (DataType).
    //!
    //!\param[in] listener  Listener to be registered.
    //!\note There is (currently) no way to unregister a
    //!      listener, so a registered DataListener must
    //!      not be destroyed before this IdcDevice
    //!      has been destroyed.
    //!\warning the methodology of this method have changed
    //!         the user does not need to register DataListeners for
    //!         each DataType individually.
    //----------------------------------------
    virtual void registerContainerListener(DataContainerListenerBase* const containerlistener);

    //========================================
    //!\brief Unregister a listener.
    //!\param[in] listener  Address of the listener object to be unregistered.
    //!\return Result of the operation.
    //----------------------------------------
    virtual StatusCode unregisterListener(DataContainerListenerBase* const containerlistener);

    //========================================
    //!\brief Register a DataStreamer to this device.
    //!
    //! Each time a message has been received by the
    //! this object, the registered streamer will be
    //! call which are listening to the received message
    //! type (DataType).
    //!
    //!\param[in] streamer  Streamer to be registered.
    //!\note There is (currently) no way to unregister a
    //!      streamer, so a registered DataStreamer must
    //!      not be destroyed before this IdcDevice
    //!      has been destroyed.
    //----------------------------------------
    virtual void registerStreamer(DataStreamer* const streamer);

    //========================================
    //!\brief Unregister a streamer.
    //!\param[in] streamer  Address of the streamer object to be unregistered.
    //!\return Result of the operation.
    //----------------------------------------
    virtual StatusCode unregisterStreamer(DataStreamer* const streamer);

    //========================================
    //!\brief Register a data package listener to this device.
    //!
    //! Each time a message has been received by the
    //! this object, the registered package listener will be
    //! call which are listening to the received message
    //! type (DataType).
    //!
    //!\param[in] listener  Package listener to be registered.
    //----------------------------------------
    virtual void registerIdcDataPackageListener(IdcDataPackageListener* const listener);

    //========================================
    //!\brief Unregister a package listener.
    //!\param[in] listener  Address of the package listener object to be unregistered.
    //!\return Result of the operation.
    //----------------------------------------
    virtual StatusCode unregisterIdcDataPackageListener(IdcDataPackageListener* const listener);

    //========================================
    //!\brief Send a command which expects no reply.
    //!\param[in] cmd  Command to be sent.
    //!\return The result of the operation.
    //----------------------------------------
    virtual StatusCode sendCommand(const CommandCBase& cmd, const SpecialExporterBase<CommandCBase>& exporter) = 0;

    //========================================
    //!\brief Send a command and wait for a reply.
    //!
    //! The command will be sent. The calling thread
    //! will sleep until a reply has been received
    //! but not longer than the number of milliseconds
    //! given in \a timeOut.
    //!
    //!\param[in]       cmd      Command to be sent.
    //!\param[in, out]  reply    The reply container for the reply to be stored into.
    //!\param[in]       timeOut  Number of milliseconds to wait for a reply.
    //!\return The result of the operation.
    //----------------------------------------
    virtual StatusCode sendCommand(const CommandCBase& cmd,
                                   const SpecialExporterBase<CommandCBase>& exporter,
                                   CommandReplyBase& reply,
                                   const boost::posix_time::time_duration timeOut
                                   = boost::posix_time::milliseconds(500))
        = 0;

protected:
    //========================================
    //!\brief Unregister all streamer and listener.
    //----------------------------------------
    void unregisterAll();

    //========================================
    //!\brief This method will be called by the receive thread
    //!       when a new DataContainer has been received completely.
    //!
    //! This class will call notifyListeners and notifyStreamers.
    //!
    //!\param[in]       dh              Header that came along with that DataContainer.
    //!\param[in, out]  is              Stream that contains the contents of
    //!                                 the DataContainer that has been received.
    //!\param[in]       containerHash   Hash for that DataContainer.
    //!\param[in, out]  importer        Importer for the DataContainer.
    //----------------------------------------
    virtual const std::shared_ptr<DataContainerBase>
    onDataReceived(const IdcDataHeader& dh,
                   std::istream& is,
                   const DataContainerBase::IdentificationKey& containerId,
                   ImporterBase*& importer)
        = 0;

    //========================================
    //!\brief Call all registered Listener listening to the received type of DataContainer.
    //!\param[in] data  DataContainer that has been received.
    //!\param[in] data  Importer for received DataContainer.
    //----------------------------------------
    virtual void notifyContainerListeners(std::shared_ptr<const DataContainerBase> dataContainer,
                                          const ImporterBase* const importer);

    //========================================
    //!\brief Call all registered Streamers listening to the received type of DataContainer.
    //!\param[in] dh       Header that came along with that DataContainer.
    //!\param[in] bodyBuf  Buffer that contains the (still serialized) body of
    //!                    the received DataContainer.
    //----------------------------------------
    virtual void notifyStreamers(const IdcDataHeader& dh, ConstCharVectorPtr bodyBuf);

    //========================================
    //!\brief Call all registered idc package listener.
    //!\param[in] data  IdcDataPackage pointer.
    //----------------------------------------
    virtual void notifyIdcDataPackageListeners(const IdcDataPackagePtr& data);

    //========================================
    //!\brief Add all registered container listeners to the list.
    //!
    //! The dataType should either be the data type from the received data block to add all listeners registered for
    //! this specific data type or DataTypeId::DataType_Unknown for the general container listeners that have not
    //! specified a data type during registration.
    //!
    //!\param[in] dataType                The data type to look for in the registered container listener map.
    //!\param[in] containerHash           The container hash to look for in the registered container listener map.
    //!\param[in/out] containerListeners  Vector with listener where the found entries will be added to.
    //----------------------------------------
    void addContainerListeners(const DataTypeId dataType,
                               const DataContainerListenerBase::HashId containerHash,
                               std::vector<DataContainerListenerBase*>& containerListeners) const;

protected:
    //========================================
    //!\brief ID of the logger for references in logging configurations.
    //----------------------------------------
    static constexpr const char* loggerId = "microvision::common::sdk::IdcDeviceBase";

protected:
    //========================================
    //!\brief Map that holds all container listener that are registered.
    //!
    //! Holds for each DataTypeId for which listener have
    //! been registered a list of those listener.
    //----------------------------------------
    RegisteredContainerListenerListMap m_registeredContainerListeners;

    //========================================
    //!\brief The list of registered streamer.
    //----------------------------------------
    StreamerList m_streamers;

    //========================================
    //!\brief The list of registered idc package listeners.
    //----------------------------------------
    IdcDataPackageListenerList m_idcPackageListeners;

    //========================================
    //!\brief Logger.
    //!
    //! Provides logging facilities for this and all derived classes.
    //----------------------------------------
    microvision::common::logging::LoggerSPtr m_logger{
        microvision::common::logging::LogManager::getInstance().createLogger(loggerId)};

    //========================================
    //!\brief Mutex protecting the listener and streamer lists.
    //----------------------------------------
    mutable Mutex m_listenersStreamersMutex;

}; // IdcDeviceBase

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
