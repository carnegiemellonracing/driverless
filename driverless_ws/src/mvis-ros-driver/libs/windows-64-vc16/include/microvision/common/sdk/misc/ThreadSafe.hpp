//==============================================================================
//! \file
//!
//! \brief Thread safe resource container.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 24, 2019
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
//! \brief Thread safe resource container.
//!
//! Make resources thread safe and access movable.
//!
//! \tparam T  any kind of value type.
//------------------------------------------------------------------------------
template<typename T>
class ThreadSafe final
{
public:
    //========================================
    //! \brief My full type.
    //----------------------------------------
    using MyType = ThreadSafe<T>;

    //========================================
    //! \brief Default mutex type.
    //----------------------------------------
    using MutexType = std::mutex;

    //========================================
    //! \brief Defaut lock guard type for wait operations.
    //----------------------------------------
    using LockType = std::unique_lock<MutexType>;

public:
    //========================================
    //! \brief Accessor to get thread-safe access over scope borders.
    //!        The destructor will unlock the mutex.
    //! \note Only creatable by ThreadSafe<T>.
    //----------------------------------------
    class Access
    {
    public:
        //========================================
        //! \brief Get parent access for creation
        //----------------------------------------
        friend class ThreadSafe<T>;

    private:
        //========================================
        //! \brief Default constructor with parent link.
        //! \param[in] value  Link to parent.
        //----------------------------------------
        explicit Access(const ThreadSafe<T>* value) : m_value{const_cast<ThreadSafe<T>*>(value)} {}

    public:
        //========================================
        //! \brief Move constructor to forward access in other scope.
        //----------------------------------------
        Access(Access&& other) : m_value{std::move(other.m_value)} { other.m_value = nullptr; }

        //========================================
        //! \brief Access is not copyable (because of thread-safe guarantee).
        //----------------------------------------
        Access(const Access& other) = delete;

        //========================================
        //! \brief Default destructor, releases access.
        //----------------------------------------
        ~Access()
        {
            if (*this)
            {
                this->m_value->m_mutex.unlock();
            }
        }

        //========================================
        //! \brief \c true if access is released, otherwise \c false.
        //! \returns Either \c true if access is released, otherwise \c false.
        //----------------------------------------
        bool operator!() { return this->m_value == nullptr; }

        //========================================
        //! \brief \c true if access isn't released, otherwise \c false.
        //! \returns Either \c true if access isn't released, otherwise \c false.
        //----------------------------------------
        operator bool() { return this->m_value != nullptr; }

        //========================================
        //! \brief Set the moved value into thread-safe scope.
        //! \param[in] value  Value to move.
        //! \returns The reference on this.
        //----------------------------------------
        Access& operator=(T&& value)
        {
            if (!(*this))
            {
                throw std::runtime_error("Access is released!");
            }
            this->m_value->m_value = std::move(value);
            return *this;
        }

        //========================================
        //! \brief Set the copied value into thread-safe scope.
        //! \param[in] value  Value to copy.
        //! \returns The reference on this.
        //----------------------------------------
        Access& operator=(const T& value)
        {
            if (!(*this))
            {
                throw std::runtime_error("Access is released!");
            }
            this->m_value->m_value = value;
            return *this;
        }

        //========================================
        //! \brief Get a pointer to the value.
        //! \returns The value pointer.
        //----------------------------------------
        T* operator->()
        {
            if (!(*this))
            {
                return nullptr;
            }
            return &this->m_value->m_value;
        }

        //========================================
        //! \brief Get a reference to the value.
        //! \returns The value reference.
        //----------------------------------------
        T& operator*()
        {
            if (!(*this))
            {
                throw std::runtime_error("Access is released!");
            }
            return this->m_value->m_value;
        }

        //========================================
        //! \brief Let the current thread waiting for the notify on this variable.
        //!        In the meantime is the access lock released.
        //----------------------------------------
        void wait()
        {
            if (!(*this))
            {
                throw std::runtime_error("Access is released!");
            }
            LockType lock{this->m_value->m_mutex, std::adopt_lock};
            this->m_value->m_changed.wait(lock);
            lock.release();
        }

        //========================================
        //! \brief Let the current thread waiting a while for the notify on this variable.
        //!        In the meantime is the access lock released.
        //! \param[in] duration  Defined how long it wait's for a signal.
        //! \returns Either \c false if running in timeout or otherwise \c true.
        //----------------------------------------
        template<typename DurationDataType, typename DurationRatio>
        bool wait_for(const std::chrono::duration<DurationDataType, DurationRatio>& duration)
        {
            if (!(*this))
            {
                throw std::runtime_error("Access is released!");
            }
            LockType lock{this->m_value->m_mutex, std::adopt_lock};
            std::cv_status result = this->m_value->m_changed.wait_for(lock, duration);
            lock.release();
            return result == std::cv_status::no_timeout;
        }

        //========================================
        //! \brief Released access.
        //----------------------------------------
        void release()
        {
            if (*this)
            {
                this->m_value->m_mutex.unlock();
                this->m_value = nullptr;
            }
        }

        //========================================
        //! \brief Released access and notify waiting threads.
        //----------------------------------------
        void releaseAndNotify()
        {
            if (*this)
            {
                this->m_value->m_mutex.unlock();
                this->m_value->notify();
                this->m_value = nullptr;
            }
        }

    private:
        //========================================
        //! \brief Parent link
        //----------------------------------------
        ThreadSafe<T>* m_value;
    };

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ThreadSafe() : m_changed{}, m_mutex{}, m_value{} {}

    //========================================
    //! \brief Move value constructor.
    //! \param[in] value  Initial value of type T.
    //----------------------------------------
    explicit ThreadSafe(T&& value) : m_changed{}, m_mutex{}, m_value{std::move(value)} {}

    //========================================
    //! \brief Copy value constructor.
    //! \param[in] value  Initial value of type T.
    //----------------------------------------
    explicit ThreadSafe(const T& value) : m_changed{}, m_mutex{}, m_value{value} {}

    //========================================
    //! \brief Deleted move constructor because of thread-safe guarantee.
    //----------------------------------------
    ThreadSafe(MyType&& other) = delete;

    //========================================
    //! \brief Deleted copy constructor because of thread-safe guarantee.
    //----------------------------------------
    ThreadSafe(const MyType& other) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ThreadSafe() = default;

public:
    //========================================
    //! \brief Set the moved value into thread-safe scope.
    //! \param[in] value  Value to move.
    //! \returns The reference on this.
    //----------------------------------------
    MyType& operator=(T&& value)
    {
        this->m_mutex.lock();
        this->m_value = std::move(value);
        this->m_mutex.unlock();
        this->notify();
        return *this;
    }

    //========================================
    //! \brief Set the copied value into thread-safe scope.
    //! \param[in] value  Value to copy.
    //! \returns The reference on this.
    //----------------------------------------
    MyType& operator=(const T& value)
    {
        this->m_mutex.lock();
        this->m_value = value;
        this->m_mutex.unlock();
        this->notify();
        return *this;
    }

    //========================================
    //! \brief Deleted move assignment, because of thread-safe guarantee.
    //----------------------------------------
    MyType& operator=(MyType&&) = delete;

    //========================================
    //! \brief Deleted copy assignment, because of thread-safe guarantee.
    //----------------------------------------
    MyType& operator=(const MyType& other) = delete;

    //========================================
    //! \brief Get a copy of the value.
    //! \returns Copy of value.
    //----------------------------------------
    operator T() { return this->getValue(); }

    //========================================
    //! \brief Get a copy of the value.
    //! \returns Copy of value.
    //----------------------------------------
    T getValue() const
    {
        LockType lock{this->m_mutex};
        return this->m_value;
    }

    //========================================
    //! \brief Get access for the calling scope.
    //! \returns An instance to get access for the calling scope.
    //----------------------------------------
    Access get()
    {
        this->m_mutex.lock();
        return Access(this);
    }

    //========================================
    //! \brief Get access for the calling scope.
    //! \returns An instance to get access for the calling scope.
    //----------------------------------------
    Access get() const
    {
        this->m_mutex.lock();
        return Access(this);
    }

    //========================================
    //! \brief Let the current thread wait for the notify on this variable.
    //!        In the meantime is the access lock released.
    //! \returns An instance to get access for the calling scope.
    //----------------------------------------
    Access wait()
    {
        LockType lock{this->m_mutex};
        this->m_changed.wait(lock);
        lock.release();
        return Access(this);
    }

    //========================================
    //! \brief Let the current thread wait a while for the notify on this variable.
    //!        In the meantime is the access lock released.
    //! \param[in] duration  Defined how long it wait's for a signal.
    //! \returns An instance to get access for the calling scope.
    //----------------------------------------
    template<typename DurationDataType, typename DurationRatio>
    Access wait_for(const std::chrono::duration<DurationDataType, DurationRatio>& duration)
    {
        LockType lock{this->m_mutex};
        this->m_changed.wait_for(lock, duration);
        lock.release();
        return Access(this);
    }

    //========================================
    //! \brief Notify when the variable has changed.
    //!        Do not call that method while the variable is locked.
    //----------------------------------------
    void notify() { this->m_changed.notify_all(); }

private:
    //========================================
    //! \brief Conditional variable to wait for notifications.
    //----------------------------------------
    std::condition_variable m_changed;

    //========================================
    //! \brief Mutex to lock resource access.
    //----------------------------------------
    mutable MutexType m_mutex;

    //========================================
    //! \brief Any value resource.
    //----------------------------------------
    T m_value;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
