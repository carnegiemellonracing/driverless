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
//! \date Jun 20, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <boost/thread.hpp>
#include <queue>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implementation of a thread-safe queue with a given capacity.
//!        Read operations block as long as the queue is empty, write operations
//!        block as long as the queue is full. Both operations can be interrupted
//!        by setting a timeout.
//------------------------------------------------------------------------------
template<typename T>
class BlockingQueue final
{
public:
    //========================================
    //! \brief Constructor
    //!
    //! \param[in] capacity  The max. number of elements in the queue.
    //----------------------------------------
    explicit BlockingQueue(const uint32_t capacity) : m_buffer(), m_capacity(capacity) {}

    //========================================
    //! \brief Destructor
    //----------------------------------------
    virtual ~BlockingQueue() = default;

    //========================================
    //! \brief Add an element to the end of the queue.
    //!
    //! This call blocks until there is space in the queue.
    //!
    //! \param[in] elem  The element to push to the queue.
    //----------------------------------------
    void push(const T& elem)
    {
        boost::unique_lock<boost::mutex> lock(m_mutex);

        // Wait for space in queue.
        m_popEvent.wait(lock, [&] { return m_buffer.size() < m_capacity; });

        // Add element.
        m_buffer.push(elem);

        // Notify one of the threads waiting for elements in queue.
        m_pushEvent.notify_one();
    }

    //========================================
    //! \brief Add an element to the end of the queue.
    //!
    //! This call blocks until there is space in the queue or the given timeout is reached.
    //!
    //! \param[in] elem           The element to push to the queue.
    //! \param[in] wait_duration  Max. time to wait for space in the queue.
    //! \return \c True if the element could be pushed, \c false if the queue is full and the call
    //!         timed out.
    //!
    //! Typical use would be (with 5 seconds timeout):
    //! \code
    //!   bool result = push<boost::posix_time::milliseconds>(elem, boost::posix_time::milliseconds(5000));
    //!   if (result)
    //!   {  success  }
    //!   else
    //!   {  timeout  }
    //! \endcode
    //----------------------------------------
    template<typename DurationType>
    bool push(const T& elem, const DurationType& wait_duration)
    {
        boost::unique_lock<boost::mutex> lock(m_mutex);

        // Wait for space in queue.
        bool result
            = m_popEvent.timed_wait<DurationType>(lock, wait_duration, [&] { return m_buffer.size() < m_capacity; });
        if (result == false)
        {
            // Timeout.
            return false;
        }

        // Add element.
        m_buffer.push(elem);

        // Notify one of the threads waiting for elements in queue.
        m_pushEvent.notify_one();

        return true; // Success.
    }

    //========================================
    //! \brief Get an element from the beginning of the queue.
    //!
    //! This call blocks until an element is available.
    //!
    //! \param[out] elem  The element retrieved from the queue.
    //----------------------------------------
    void pop(T& elem)
    {
        boost::unique_lock<boost::mutex> lock(m_mutex);

        // Wait for elements in queue.
        m_pushEvent.wait(lock, [&] { return m_buffer.empty() == false; });

        // Fetch element.
        elem = m_buffer.front();
        m_buffer.pop();

        // Notify one of the threads waiting for space in the queue.
        m_popEvent.notify_one();
    }

    //========================================
    //! \brief Get an element from the beginning of the queue.
    //!
    //! This call blocks until an element is available or the given timeout is reached.
    //!
    //! \param[out] elem           The element retrieved from the queue (undefined in case of timeout).
    //! \param[in]  wait_duration  Max. time to wait for elements in the queue.
    //! \return \c True if an element could be fetched, \c false if the queue is empty and the call
    //!         timed out.
    //!
    //! Typical use would be (with 5 seconds timeout):
    //! \code
    //!   bool result = pop<boost::posix_time::milliseconds>(elem, boost::posix_time::milliseconds(5000));
    //!   if (result)
    //!   {  success  }
    //!   else
    //!   {  timeout  }
    //! \endcode
    //----------------------------------------
    template<typename DurationType>
    bool pop(T& elem, const DurationType& wait_duration)
    {
        boost::unique_lock<boost::mutex> lock(m_mutex);

        // Wait for elements in queue.
        bool result
            = m_pushEvent.timed_wait<DurationType>(lock, wait_duration, [&] { return m_buffer.empty() == false; });
        if (result == false)
        {
            // Timeout.
            return false;
        }

        // Fetch element.
        elem = m_buffer.front();
        m_buffer.pop();

        // Notify one of the threads waiting for space in the queue.
        m_popEvent.notify_one();

        return true; // Success.
    }

private:
    boost::mutex m_mutex; // mutex for locking the queue
    boost::condition_variable m_pushEvent; // event signaling there is an element in the queue
    boost::condition_variable m_popEvent; // event signaling there is space in the queue
    std::queue<T> m_buffer; // the queue
    size_t m_capacity; // the max. number of elements in the queue
}; // BlockingQueue

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
