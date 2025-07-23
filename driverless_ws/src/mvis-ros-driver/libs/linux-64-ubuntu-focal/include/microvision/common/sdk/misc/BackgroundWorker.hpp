//==============================================================================
//! \file
//!
//! \brief Background worker for easier multithread handling.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 28, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/ThreadRunner.hpp>

#include <microvision/common/logging/logging.hpp>

#include <functional>
#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Background worker for easier multithread handling.
//!
//! Helper class to wrap ThreadRunner and handle thread status.
//! Exceptions will be handled and the last one will be stored.
//! Observer methods to easier handle shared data access.
//------------------------------------------------------------------------------
class BackgroundWorker
{
public:
    //========================================
    //! \brief Function definition which will call in background thread.
    //! \param[in, out] worker  Background worker which called the function.
    //! \returns Either \c true if the thread should keep alive or otherwise \c false.
    //----------------------------------------
    using WorkFunction = std::function<bool(BackgroundWorker& worker)>;

    //========================================
    //! \brief Default time which the observer will wait for a notification by another thread.
    //----------------------------------------
    static constexpr uint32_t defaultWaitTimeInMs{500U};

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BackgroundWorker";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr getLogger();

public:
    //========================================
    //! \brief Observe thread safe list resource.
    //!
    //!  If something is there or will be added while observing all queued entries will be gathered.
    //!
    //! \tparam         ListType        Type of list resource.
    //! \param[in, out] access          Thread safe list resource.
    //! \param[in, out] data            Output list which will be filled with the queued entries.
    //! \param[in]      waitTimeInMs    (Optional) How long do you want to wait for changes in milliseconds. (Default: 500 Ms)
    //----------------------------------------
    template<typename ListType>
    static void observeOnce(typename ThreadSafe<ListType>::Access&& access,
                            ListType& data,
                            const uint32_t waitTimeInMs = defaultWaitTimeInMs);

    //========================================
    //! \brief Observe thread-safe list resource.
    //!
    //!  If something is there or will be added while observing,
    //!  get the first queued entry.
    //!
    //! \tparam         ValueType       Type of value which is in list.
    //! \tparam         ListType        (Optional) Type of list resource which contains value type. (Default: std::list<ValueType>)
    //! \param[in, out] access          Thread safe list resource.
    //! \param[in]      defaultValue    (Optional) Default value which will returns if no entry get. (Default: Empty construction of ValueType)
    //! \param[in]      waitTimeInMs    (Optional) How long do you want to wait for changes in milliseconds. (Default: 500 Ms)
    //! \returns Either the first value of thread safe list resource or otherwise the default value.
    //----------------------------------------
    template<typename ValueType, typename ListType = std::list<ValueType>>
    static ValueType observeOnce(typename ThreadSafe<ListType>::Access&& access,
                                 const ValueType& defaultValue = ValueType{},
                                 const uint32_t waitTimeInMs   = defaultWaitTimeInMs);

public:
    //========================================
    //! \brief Construct an background worker by his main method.
    //! \param[in] work  Main method of background thread.
    //----------------------------------------
    BackgroundWorker(const WorkFunction& work);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~BackgroundWorker();

public:
    //========================================
    //! \brief Checks if thread is running.
    //! \returns Either \c true if thread runs or otherwise \c false.
    //----------------------------------------
    bool isRunning() const;

    //========================================
    //! \brief Get all caught exceptions.
    //! \returns List of exception pointers.
    //----------------------------------------
    std::vector<std::exception_ptr> getErrors() const;

    //========================================
    //! \brief Get the last caught exception.
    //! \returns Either last caught exception or otherwise nullptr.
    //----------------------------------------
    std::exception_ptr getLastError() const;

    //========================================
    //! \brief Set current exception as last error.
    //! \param[in] exceptionPtr  Exception pointer, as example \see std::current_exception()
    //----------------------------------------
    void setLastError(const std::exception_ptr& exceptionPtr);

    //========================================
    //! \brief Set error code as caught system exception.
    //! \param[in] errorCode  Error code of system exception.
    //----------------------------------------
    void setLastError(const int errorCode);

    //========================================
    //! \brief Set error code and message as caught system exception.
    //! \param[in] errorCode  Error code of system exception.
    //! \param[in] message    Error message of system exception.
    //----------------------------------------
    void setLastError(const int errorCode, const std::string& message);

public:
    //========================================
    //! \brief Start thread.
    //----------------------------------------
    void start();

    //========================================
    //! \brief Stop thread.
    //!
    //! Set state to stopping and join.
    //!
    //! \note Please check if it is still running in main function for the cancel implementation.
    //----------------------------------------
    void stop();

    //========================================
    //! \brief Wait until thread is finished.
    //----------------------------------------
    void join();

private:
    //========================================
    //! \brief Main method to wrap ThreadRunner.
    //----------------------------------------
    void main();

private:
    //========================================
    //! \brief Background worker main method.
    //----------------------------------------
    WorkFunction m_work;

    //========================================
    //! \brief Wrapped ThreadRunner.
    //----------------------------------------
    ThreadRunner m_thread;

    //========================================
    //! \brief Caught exceptions in thread.
    //----------------------------------------
    ThreadSafe<std::vector<std::exception_ptr>> m_errors;

}; // class BackgroundWorker

//==============================================================================
//! \brief Nullable BackgroundWorker pointer.
//------------------------------------------------------------------------------
using BackgroundWorkerUPtr = std::unique_ptr<BackgroundWorker>;

//==============================================================================

template<typename ListType>
void BackgroundWorker::observeOnce(typename ThreadSafe<ListType>::Access&& access,
                                   ListType& data,
                                   const uint32_t waitTimeInMs)
{
    if (access->empty())
    {
        access.wait_for(std::chrono::milliseconds(waitTimeInMs));
    }

    while (!access->empty())
    {
        data.push_back(access->front());
        access->pop_front();
    }

    if (!data.empty())
    {
        access.releaseAndNotify();
    }
}

//==============================================================================

template<typename ValueType, typename ListType>
ValueType BackgroundWorker::observeOnce(typename ThreadSafe<ListType>::Access&& access,
                                        const ValueType& defaultValue,
                                        const uint32_t waitTimeInMs)
{
    if (access->empty())
    {
        access.wait_for(std::chrono::milliseconds(waitTimeInMs));
    }

    if (!access->empty())
    {
        ValueType value = access->front();
        access->pop_front();
        access.releaseAndNotify();
        return value;
    }
    return defaultValue;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
