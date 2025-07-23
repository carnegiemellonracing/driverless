//==============================================================================
//! \file
//! \brief Class for handling logger backends.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/backends/LoggerBackend.hpp>
#include <microvision/common/logging/Configuration.hpp>
#include <microvision/common/logging/Format.hpp>
#include <microvision/common/logging/LoggingExport.hpp>

#include <map>
#include <algorithm>
#include <memory>
#include <mutex>

//==============================================================================

#ifdef _WIN32
#    pragma warning(disable : 4251)
// class 'xxx' needs to have dll - interface to be used by clients of class 'yyy'
#endif

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief Central class for accessing logger backends.
//------------------------------------------------------------------------------
class LOGGING_EXPORT LoggerBackendManager final
{
public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    LoggerBackendManager() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~LoggerBackendManager();

public:
    //========================================
    //! \brief Register a backend at the factory.
    //!
    //! \param[in] backend  The backend to register.
    //! \return  \c true if the registration was successful, \c false otherwise.
    //----------------------------------------
    bool registerBackend(LoggerBackendSPtr backend);

    //========================================
    //! \brief Get a backend by its ID.
    //!
    //! \param[in] backendId  ID of the backend to look for.
    //! \return  The backend if the given ID was found, or an empty shared pointer (\c nullptr) otherwise.
    //----------------------------------------
    LoggerBackendSPtr getBackendById(const std::string& backendId);

    //========================================
    //! \brief Waits until all backends have processed all log messages.
    //!
    //! \param[in] maxFlushTimeMilliseconds  Max. time in milliseconds to wait until.
    //! \return \c true if all log messages have been processed, or \c false if a timeout occurred.
    //----------------------------------------
    void flushBackends(const uint32_t maxFlushTimeMilliseconds = 100) const;

private:
    using Mutex      = std::mutex;
    using MutexGuard = std::lock_guard<Mutex>;
    using BackendMap = std::map<std::string, LoggerBackendSPtr>;

private:
    mutable Mutex m_mutex{};
    BackendMap m_backends;
}; // LoggerBackendManager

//==============================================================================

using LoggerBackendManagerSPtr = std::shared_ptr<LoggerBackendManager>;

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
