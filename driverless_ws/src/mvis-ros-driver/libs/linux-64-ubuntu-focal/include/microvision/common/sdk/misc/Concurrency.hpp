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
//! \date May 09, 2012
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class that provides a mutex mechanism for it heir.
//! \date Mai 9, 2012
//------------------------------------------------------------------------------
class Concurrency final
{
public:
    //========================================
    //! \brief Default constructor of Concurrency.
    //----------------------------------------
    Concurrency() : m_condition(), m_mutex(), m_locked(false) {}

    //========================================
    //! \brief Destructor, doing nothing special.
    //----------------------------------------
    virtual ~Concurrency() {}

public:
    //========================================
    //! \brief Lock the mutex.
    //----------------------------------------
    virtual void lock() const
    {
        boost::lock_guard<boost::mutex> lockGuard(m_mutex);
        m_locked = true;
    }

    //========================================
    //! \brief Unlock the mutex.
    //----------------------------------------
    virtual void unlock() const
    {
        m_locked = false;
        m_condition.notify_all();
    }

    //========================================
    //! \brief Wait till the mutex is free.
    //!
    //! The method will return immediately if
    //! the mutex is not locked currently.
    //----------------------------------------
    virtual void wait() const
    {
        boost::unique_lock<boost::mutex> lockGuard(m_mutex);
        while (isLocked())
            m_condition.wait(lockGuard);
    }

    //========================================
    //! \brief Check whether the mutex is locked.
    //! \return Whether the mutex is locked.
    //! \retval true The mutex is still locked.
    //! \retval false The mutex is free.
    //----------------------------------------
    virtual bool isLocked() const { return m_locked; }

protected:
    //========================================
    //! \brief Condition variable used by the mutex mechanism.
    //----------------------------------------
    mutable boost::condition_variable m_condition;

    //========================================
    //! \brief Mutex.
    //----------------------------------------
    mutable boost::mutex m_mutex;

    //========================================
    //! \brief Variable to hold the status of the mutex.
    //----------------------------------------
    mutable bool m_locked;
}; // Concurrency

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
