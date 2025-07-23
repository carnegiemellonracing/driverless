//==============================================================================
//! \file
//!
//! \brief Thread handling context for sharing resources.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 19, 2021
//------------------------------------------------------------------------------
#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <mutex>
#include <memory>
#include <condition_variable>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Context to lock thread until resource available.
//! \tparam MutexType  the type of the mutex to lock. The type must meet the BasicLockable requirements.
//------------------------------------------------------------------------------
template<typename MutexType>
class ThreadShareContext
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ThreadShareContext() = default;

    //========================================
    //! \brief Disable move constructor (because of thread-safe guarantee).
    //----------------------------------------
    ThreadShareContext(ThreadShareContext<MutexType>&&) = delete;

    //========================================
    //! \brief Disable copy constructor (because of thread-safe guarantee).
    //----------------------------------------
    ThreadShareContext(const ThreadShareContext<MutexType>&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~ThreadShareContext() = default;

    //========================================
    //! \brief Disable move assignment operator (because of thread-safe guarantee).
    //----------------------------------------
    ThreadShareContext& operator=(ThreadShareContext<MutexType>&&) = delete;

    //========================================
    //! \brief Disable copy assignment operator (because of thread-safe guarantee).
    //----------------------------------------
    ThreadShareContext& operator=(const ThreadShareContext<MutexType>&) = delete;

public:
    //========================================
    //! \brief Mutex to lock resource access.
    //----------------------------------------
    mutable MutexType shareHandler{};

}; // class ThreadShareContext

//==============================================================================
//! \brief Pointer to an instance of \a ThreadShareContext for sharing context.
//! \tparam MutexType  the type of the mutex to lock. The type must meet the BasicLockable requirements.
//------------------------------------------------------------------------------
template<typename MutexType>
using ThreadShareContextPtr = std::shared_ptr<ThreadShareContext<MutexType>>;

//==============================================================================
//! \brief Context to lock thread until signal received.
//! \note Required share context for lock.
//! \tparam MutexType  the type of the mutex to lock. The type must meet the BasicLockable requirements.
//------------------------------------------------------------------------------
template<typename MutexType>
class ThreadSyncContext
{
public:
    //========================================
    //! \brief Defaut lock guard type for wait operations.
    //----------------------------------------
    using LockType = std::unique_lock<MutexType>;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ThreadSyncContext() = default;

    //========================================
    //! \brief Disable move constructor (because of thread-safe guarantee).
    //----------------------------------------
    ThreadSyncContext(ThreadSyncContext<MutexType>&&) = delete;

    //========================================
    //! \brief Disable copy constructor (because of thread-safe guarantee).
    //----------------------------------------
    ThreadSyncContext(const ThreadSyncContext<MutexType>&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~ThreadSyncContext() = default;

    //========================================
    //! \brief Disable move assignment operator (because of thread-safe guarantee).
    //----------------------------------------
    ThreadSyncContext& operator=(ThreadSyncContext<MutexType>&&) = delete;

    //========================================
    //! \brief Disable copy assignment operator (because of thread-safe guarantee).
    //----------------------------------------
    ThreadSyncContext& operator=(const ThreadSyncContext<MutexType>&) = delete;

public:
    //========================================
    //! \brief Conditional variable to wait for signal.
    //----------------------------------------
    std::condition_variable syncHandler{};
}; // class ThreadSyncContext

//==============================================================================
//! \brief Pointer to any instance of \a ThreadSyncContext for sharing context.
//! \tparam MutexType  the type of the mutex to lock. The type must meet the BasicLockable requirements.
//------------------------------------------------------------------------------
template<typename MutexType>
using ThreadSyncContextPtr = std::shared_ptr<ThreadSyncContext<MutexType>>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
