//==============================================================================
//! \file
//!
//! \brief Sync shared resources/stats between threads.
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
#include <microvision/common/sdk/misc/ThreadContext.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Controller to sync shared resources/stats between threads.
//------------------------------------------------------------------------------
class ThreadSync final
{
public:
    //========================================
    //! \brief Default mutex type.
    //----------------------------------------
    using MutexType = std::mutex;

    //========================================
    //! \brief Type of \a ThreadShareContext with \a MutexType.
    //----------------------------------------
    using ShareContextType = ThreadShareContext<MutexType>;

    //========================================
    //! \brief Pointer type of \a ThreadShareContext with \a MutexType.
    //----------------------------------------
    using ShareContextPtrType = ThreadShareContextPtr<MutexType>;

    //========================================
    //! \brief Type of \a ThreadSyncContext with \a MutexType.
    //----------------------------------------
    using SyncContextType = ThreadSyncContext<MutexType>;

    //========================================
    //! \brief Pointer type of \a ThreadSyncContextPtr with \a MutexType.
    //----------------------------------------
    using SyncContextPtrType = ThreadSyncContextPtr<MutexType>;

public:
    //========================================
    //! \brief Default constructor.
    //! \note Will make new share and sync context.
    //----------------------------------------
    ThreadSync() : ThreadSync(nullptr, nullptr) {}

    //========================================
    //! \brief Construct instance with thread share context to share thread lock handle.
    //! \param[in] shareContext  Thread share context to share thread lock handle.
    //! \note Will make new snyc context and if \a shareContext is nullptr.
    //----------------------------------------
    explicit ThreadSync(const ShareContextPtrType& shareContext) : ThreadSync(shareContext, nullptr) {}

    //========================================
    //! \brief Construct instance with thread snyc context to share thread signale handle.
    //! \param[in] syncContext  Thread sync context to share thread signal handle.
    //! \note Will make new share context and if \a syncContext is nullptr.
    //----------------------------------------
    explicit ThreadSync(const SyncContextPtrType& syncContext) : ThreadSync(nullptr, syncContext) {}

    //========================================
    //! \brief Construct instance with thread share/sync context to share thread lock/signal handle.
    //! \param[in] shareContext  Pointer to thread share context to share thread lock handle.
    //! \param[in] syncContext  Pointer thread sync context to share thread signal handle.
    //! \note Will make new context if \a shareContext or \a syncContext is nullptr.
    //----------------------------------------
    ThreadSync(const ShareContextPtrType& shareContext, const SyncContextPtrType& syncContext)
      : m_share{shareContext}, m_sync{syncContext}
    {
        if (!this->m_share)
        {
            this->m_share = std::make_shared<ShareContextType>();
        }
        if (!this->m_sync)
        {
            this->m_sync = std::make_shared<SyncContextType>();
        }
    }

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Other instance of \a ThreadSync to move.
    //----------------------------------------
    ThreadSync(ThreadSync&& other) : m_share{std::move(other.m_share)}, m_sync{std::move(other.m_sync)} {}

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of \a ThreadSync to copy.
    //----------------------------------------
    ThreadSync(const ThreadSync& other) : m_share{other.m_share}, m_sync{other.m_sync} {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ThreadSync() = default;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Other instance of \a ThreadSync to move.
    //----------------------------------------
    ThreadSync& operator=(ThreadSync&& other)
    {
        this->m_share = std::move(other.m_share);
        this->m_sync  = std::move(other.m_sync);
        return *this;
    }

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of \a ThreadSync to copy.
    //----------------------------------------
    ThreadSync& operator=(const ThreadSync& other)
    {
        this->m_share = other.m_share;
        this->m_sync  = other.m_sync;
        return *this;
    }

    //========================================
    //! \brief Let the current thread wait for the notification.
    //----------------------------------------
    void wait()
    {
        typename SyncContextType::LockType lock{this->m_share->shareHandler};
        this->m_sync->syncHandler.wait(lock);
    }

    //========================================
    //! \brief Let the current thread wait a while for the notification.
    //! \param[in] duration  Defines how long it is waiting for a signal.
    //! \returns Either \c true if wait interrupted by notification or otherwise \c false.
    //----------------------------------------
    template<typename DurationDataType, typename DurationRatio>
    bool waitFor(const std::chrono::duration<DurationDataType, DurationRatio>& duration)
    {
        typename SyncContextType::LockType lock{this->m_share->shareHandler};
        const auto result = this->m_sync->syncHandler.wait_for(lock, duration);
        return result == std::cv_status::no_timeout;
    }

    //========================================
    //! \brief Push notification (signal) to interrupt the waiting threads.
    //----------------------------------------
    void notify() { this->m_sync->syncHandler.notify_all(); }

private:
    //========================================
    //! \brief Context to lock the thread.
    //----------------------------------------
    ShareContextPtrType m_share;

    //========================================
    //! \brief Context to wait for interrupt.
    //----------------------------------------
    SyncContextPtrType m_sync;

}; // class ThreadSync

//==============================================================================
//! \brief Pointer to an instance of \a ThreadSync for sharing thread resources/stats.
//------------------------------------------------------------------------------
using ThreadSyncPtr = std::shared_ptr<ThreadSync>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
