//==============================================================================
//! \file
//!
//!\brief Interface for general io network resources.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 30, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/Uri.hpp>

#include <exception>
#include <mutex>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Interface for general io network resources.
//!
//! Implement this interface to provide general network resource handling.
//! Will implemented by DataPackageNetworkReceiver and DataPackageNetworkSender for example.
//!
//! \note Inherit this interface as virtual.
//------------------------------------------------------------------------------
class NetworkBase
{
public:
    //========================================
    //! \brief Mutex type for thread-safe implementations.
    //----------------------------------------
    using Mutex = std::recursive_mutex;

    //========================================
    //! \brief Lock guard type for thread-safe implementations.
    //----------------------------------------
    using LockGuard = std::lock_guard<Mutex>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~NetworkBase() = default;

public: // getter
    //========================================
    //! \brief Get the source/destination Uri.
    //! \return Describing source/destination Uri of network.
    //----------------------------------------
    virtual Uri getUri() const = 0;

    //========================================
    //! \brief Get the last error which is caught in network io thread.
    //! \returns Exception pointer if error caught, otherwise nullptr.
    //----------------------------------------
    virtual std::exception_ptr getLastError() const = 0;

public:
    //========================================
    //! \brief Checks that a connection is established.
    //! \returns Either \c true if connection is established or otherwise \c false.
    //----------------------------------------
    virtual bool isConnected() const = 0;

    //========================================
    //! \brief Checks that a connection is established and packages are still processed.
    //! \returns Either \c true if connection is established or work is ongoing on received data or otherwise \c false.
    //----------------------------------------
    virtual bool isWorking() = 0;

public:
    //========================================
    //! \brief Establish a connection to the network resource.
    //----------------------------------------
    virtual void connect() = 0;

    //========================================
    //! \brief Disconnect from the network resource.
    //----------------------------------------
    virtual void disconnect() = 0;
}; // class NetworkBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
