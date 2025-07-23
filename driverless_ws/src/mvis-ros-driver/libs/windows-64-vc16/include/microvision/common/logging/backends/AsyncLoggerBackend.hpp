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

#include <microvision/common/logging/backends/LoggerBackend.hpp>
#include <microvision/common/logging/backends/LogMessageQueue.hpp>

#include <microvision/common/logging/LogLevel.hpp>
#include <microvision/common/logging/Message.hpp>
#include <microvision/common/logging/LoggingExport.hpp>

#include <cstdint>

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

class LOGGING_EXPORT AsyncLoggerBackend : public LoggerBackend
{
public:
    static const uint32_t logMessageDefaultQueueSize;

public: // constructors, destructors
    //========================================
    //! \brief Constructor.
    //!
    //! \param[in] backendId  The unique ID of this backend.
    //----------------------------------------
    AsyncLoggerBackend(const std::string& backendId);

    //========================================
    //! \brief Destructor.
    //!
    //! \note Derived classes must call stopThread() in their destructor.
    //----------------------------------------
    virtual ~AsyncLoggerBackend();

public:
    //========================================
    //! \brief Get the size of the log message queue.
    //!
    //! \return the max. number of entries the log message queue can hold
    //----------------------------------------
    std::size_t getLogMessageQueueSize() const;

    //========================================
    //! \brief Get the timeout for the log message queue.
    //!
    //! \return the timeout in milliseconds that the thread sending the log message waits at most if the queue is full.
    //----------------------------------------
    uint32_t getLogMessageQueueTimeoutMilliseconds() const;

public:
    //========================================
    //! \brief Configure this backend from an XML node.
    //!
    //! \param[in] xmlNode         XML node containing the configuration.
    //! \param[in] suppressErrors  If set to \c true output of error messages shall be suppressed, otherwise they shall
    //!                            be printed to stderr.
    //! \return  \c true if the configuration was successful, \c false otherwise.
    //!
    //! \note Sample configuration inside the XML node:
    //! \code
    //! <Backend id="microvision::common::logging::FileLoggerBackend"
    //!          format="%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n">
    //!     <Mode>Block</Mode>
    //!     <QueueSize>1024</QueueSize>
    //!     <Timeout>100</Timeout>
    //!     <Path>microvision.log</Path>
    //! </Backend>
    //! \endcode
    //! Mode parameter can be one of \c Block, \c Timed, \c Drop, or \c Bypass.
    //----------------------------------------
    virtual bool configure(const tinyxml2::XMLElement* const xmlNode, const bool suppressErrors);

    //========================================
    //! \brief Log a message asynchronously.
    //!
    //! \param[in] loggerId  ID of the logger which sent the message.
    //! \param[in] level     Log level as determined by the logger.
    //! \param[in] msg       The message to log.
    //!
    //! \note If the backend is configured in direct mode this method is used to log a message synchronously.
    //----------------------------------------
    virtual void logAsync(const std::string& loggerId, const LogLevel& level, const MessageSPtr msg) = 0;

    //========================================
    //! \brief Waits until all log messages are processed.
    //!
    //! \param[in] maxFlushTimeMilliseconds  Max. time in milliseconds to wait until all log messages are processed.
    //! \return \c true if all log messages have been processed, or \c false if a timeout occurred.
    //----------------------------------------
    bool flush(const uint32_t maxFlushTimeMilliseconds = 100) const final;

    //========================================
    //! \brief Log a message.
    //!
    //! \param[in] loggerId  ID of the logger which sent the message.
    //! \param[in] level     Log level as determined by the logger.
    //! \param[in] msg       The message to log.
    //!
    //! \note This method is used internally for asynchronous logging. Thus, it is not allowed to override this method.
    //----------------------------------------
    void log(const std::string& loggerId, const LogLevel& level, const MessageSPtr msg) final;

protected:
    using Mutex      = std::mutex;
    using MutexGuard = std::unique_lock<Mutex>;

protected:
    //========================================
    //! \brief Empties the internal message queue and stops the internal thread.
    //!
    //! \return True iff the internal thread was running.
    //!
    //! \note Derived classes must call this function in their destructor.
    //----------------------------------------
    bool stopThread();

    //========================================
    //! \brief Returns true iff the internal thread is running.
    //----------------------------------------
    bool isThreadRunning() const;

protected:
    mutable Mutex m_mutex{};

private:
    using Thread     = std::thread;
    using ThreadSPtr = std::shared_ptr<Thread>;

private:
    void reset();

    void threadMain();

private:
    mutable Mutex m_threadMutex{};
    LogMessageQueueSPtr m_queue{nullptr}; //!< Queue for logging asynchronously.
    ThreadSPtr m_thread{nullptr}; //!< Thread handling asynchronous log messages.
    bool m_handlingMessage{false}; //!< Flag whether currently handling a message.
    bool m_bypassQueue{false}; //!< Flag for synchronous logging (true, without thread and queue) or
        //!< asynchronous logging (false).
    uint32_t m_messageQueueWaitTime{LogMessageQueue::infiniteMilliseconds};
    //!< max. time to wait for space in log message queue
    uint32_t m_messageQueueSize{logMessageDefaultQueueSize};
    //!< max. size of log message queue

}; // AsyncLoggerBackend

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
