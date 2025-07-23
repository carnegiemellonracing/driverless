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
//! \date Feb 10, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/sdk/listener/DataPackageListener.hpp>

#include <microvision/common/logging/logging.hpp>

#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface which provide emit functionality for data package listeners.
//!
//! A class implementing this interface can act as a subject notifying
//! registered listeners  whenever a new data package is received or processed
//! by the device derived from this class. The listener then can further process the data.
//! See observer pattern (Subject).
//! Also have look at the listener demos.
//! /sa DataPackageListener
//------------------------------------------------------------------------------
class Emitter
{
public:
    //========================================
    //! \brief Nullable DataPackageListener pointer.
    //----------------------------------------
    using DataPackageListenerPtr = std::shared_ptr<DataPackageListener>;

    //========================================
    //! \brief List type of DataPackageListenerPtr.
    //----------------------------------------
    using DataPackageListenerList = std::list<DataPackageListenerPtr>;

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::Emitter";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    Emitter();

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    Emitter(const Emitter& other);

    //========================================
    //! \brief Disabled move constructor.
    //----------------------------------------
    Emitter(Emitter&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~Emitter();

public:
    //========================================
    //! \brief Register listener at subject to receive data packages.
    //! \param[in] listener  Pointer to data package listener.
    //----------------------------------------
    void registerDataPackageListener(const DataPackageListenerPtr& listener);

    //========================================
    //! \brief Unregister listener from subject to avoid receiving data packages.
    //! \param[in] listener  Pointer to data package listener.
    //----------------------------------------
    void unregisterDataPackageListener(const DataPackageListenerPtr& listener);

public:
    //========================================
    //! \brief Notify data package listeners about a new data package.
    //!
    //! A new data package can be read from network or file stream or put into the emitter in other ways
    //! depending on the implementation of the emitter.
    //! Listeners will be notified regardless of the source of the package.
    //!
    //! \param[in] dataPackage  Pointer to data package of which the registered listeners will be notified.
    //! \return Either \c true if listeners are triggered or otherwise \c false.
    //----------------------------------------
    bool notifyDataPackageListeners(const DataPackagePtr& dataPackage);

private:
    //========================================
    //! \brief List of registered data package listeners.
    //----------------------------------------
    ThreadSafe<DataPackageListenerList> m_dataPackageListeners;
}; // class Emitter

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
