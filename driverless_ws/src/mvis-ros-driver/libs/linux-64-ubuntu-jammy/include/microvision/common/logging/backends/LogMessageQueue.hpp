//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 06, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/LogLevel.hpp>
#include <microvision/common/logging/Message.hpp>
#include <microvision/common/logging/LoggingExport.hpp>

#include <condition_variable>
#include <queue>
#include <cstdint>

// undefine already defined max
#ifdef max
#    undef max
#endif

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

class LOGGING_EXPORT LogMessageQueue
{
public:
    static const uint32_t infiniteMilliseconds;
    // ~ 50 days, should not overflow with std::chrono::steady_clock::now().

public:
    //========================================
    //! \brief Constructor.
    //!
    //! \param[in] capacity  the max. number of entries the queue can hold.
    //----------------------------------------
    explicit LogMessageQueue(const std::size_t capacity) : m_buffer(), m_capacity(capacity) {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~LogMessageQueue() = default;

public:
    //========================================
    //! \brief Get the queue capacity.
    //!
    //! \return the max. number of entries the queue can hold.
    //----------------------------------------
    std::size_t getCapacity() const { return m_capacity; }

    //========================================
    //! \brief Get the timeout for the queue.
    //!
    //! \return the timeout in milliseconds that the thread sending the log message waits at most if the queue is full.
    //----------------------------------------
    uint32_t getWaitTimeMilliseconds() const { return static_cast<uint32_t>(m_pushWaitTime.count()); }

    //========================================
    //! \brief Set the timeout for the queue.
    //!
    //! \param[in] timeout  the timeout in milliseconds that the thread sending the log message waits at most
    //!                     if the queue is full.
    //----------------------------------------
    void setWaitTimeMilliseconds(const uint32_t timeout) { m_pushWaitTime = std::chrono::milliseconds(timeout); }

    //========================================
    //! \brief Checks whether the queue is empty.
    //!
    //! \return \c true if the queue is empty, \c false otherwise.
    //----------------------------------------
    bool isEmpty() const { return m_buffer.empty(); }

    //========================================
    //! \brief Clears the queue.
    //----------------------------------------
    void clear();

public:
    //========================================
    //! \brief Adds an entry to the queue.
    //!
    //! \param[in,out] lock  Lock used to protect the queue against multithreaded access.
    //! \param[in] loggerId  ID of the logger which sent the message.
    //! \param[in] level     Log level as determined by the logger.
    //! \param[in] msg       The message to log.
    //! \return \c true if the entry was added, \c false if a timeout occurred while the queue is full.
    //----------------------------------------
    bool
    push(std::unique_lock<std::mutex>& lock, const std::string& loggerId, const LogLevel& level, const MessageSPtr msg);

    //========================================
    //! \brief Retrieves an entry from the queue.
    //!
    //! \param[in,out] lock   Lock used to protect the queue against multithreaded access.
    //! \param[out] loggerId  ID of the logger which sent the message.
    //! \param[out] level     Log level as determined by the logger.
    //! \param[out] msg       The message to log.
    //----------------------------------------
    void pop(std::unique_lock<std::mutex>& lock, std::string& loggerId, LogLevel& level, MessageSPtr& msg);

private:
    struct Entry
    {
        Entry(const std::string& loggerId, LogLevel level, const MessageSPtr& msg)
          : m_loggerId(loggerId), m_level(level), m_msg(msg)
        {}

    public:
        std::string m_loggerId;
        LogLevel m_level;
        MessageSPtr m_msg;
    }; // Entry

private:
    std::condition_variable m_pushEvent; // Event signaling there is an element in the queue.
    std::condition_variable m_popEvent; // Event signaling there is space in the queue.
    std::queue<Entry> m_buffer; // The queue.
    std::size_t m_capacity; // Max. number of elements in the queue.
    std::chrono::milliseconds m_pushWaitTime{infiniteMilliseconds};
    // Max. number of milliseconds to wait when pushing an element and the
    // queue is full.
}; // LogMessageQueue

//==============================================================================

using LogMessageQueueSPtr = std::shared_ptr<LogMessageQueue>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
