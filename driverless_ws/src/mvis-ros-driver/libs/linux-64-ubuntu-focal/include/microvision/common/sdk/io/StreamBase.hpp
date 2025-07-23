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
//! \date Aug 19, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/Uri.hpp>

#include <microvision/common/sdk/config/Configuration.hpp>
#include <microvision/common/sdk/config/Configurable.hpp>

#include <microvision/common/logging/logging.hpp>

#include <iostream>
#include <memory>
#include <mutex>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Interface for general io stream resources.
//!
//! Implement this interface to provide general stream resource handling.
//!
//! \note Inherit this interface as virtual.
//! \extends Configurable
//------------------------------------------------------------------------------
class StreamBase : public Configurable
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

    //========================================
    //! \brief std::iostream pointer type for delayed initialization.
    //----------------------------------------
    using IoStreamPtr = std::unique_ptr<std::iostream>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~StreamBase() = default;

public:
    //========================================
    //! \brief Get pointer to configuration which is used to define/filter/prepare input.
    //! \return Pointer to an instance of Configuration.
    //----------------------------------------
    virtual ConfigurationPtr getConfiguration() const = 0;

    //========================================
    //! \brief Set pointer to configuration which is used to define/filter/prepare input.
    //!
    //! \param[in] configuration  Pointer to an instance of Configuration.
    //! \return Either \c true if the configuration is supported by implementation or otherwise \c false.
    //! \note If the configuration is not supported by implementation it will not change the current value.
    //!       However, if \a configuration is \c nullptr the configuration of NetworkInterface will be reset.
    //----------------------------------------
    virtual bool setConfiguration(const ConfigurationPtr& configuration) = 0;

public:
    //========================================
    //! \brief Request resource access.
    //----------------------------------------
    virtual bool open() { return this->open(nullptr); }

    //========================================
    //! \brief Release and close resources.
    //----------------------------------------
    virtual void close() { this->release(); }

public:
    //========================================
    //! \brief Get the source/destination Uri.
    //! \return Describing source/destination Uri of stream.
    //----------------------------------------
    virtual const Uri& getUri() const = 0;

    //========================================
    //! \brief Checks if the stream is accessible
    //! \attention That does not check if the stream is in failed state because of
    //!             the possible to start a new read process with clearing the flags.
    //! \return Either \c true if the resource is in good condition, otherwise \c false.
    //----------------------------------------
    virtual bool isGood() const = 0;

    //========================================
    //! \brief Checks if the stream is not accessible or is unrecoverable.
    //! \return Either \c true if the resource is in bad condition, otherwise \c false.
    //----------------------------------------
    virtual bool isBad() const = 0;

    //========================================
    //! \brief Checks if the stream is not accessible or is unrecoverable or EOF.
    //! \return Either \c true if the resource is in bad or EOF condition, otherwise \c false.
    //----------------------------------------
    virtual bool isEof() const = 0;

    //========================================
    //! \brief Request resource access and takeover of the stream ownership.
    //! \param[in] ioStream  Resource Stream handle.
    //----------------------------------------
    virtual bool open(IoStreamPtr&& ioStream) = 0;

    //========================================
    //! \brief Release resources and stream ownership.
    //! \returns Get stream ownership back.
    //----------------------------------------
    virtual IoStreamPtr release() = 0;

    //========================================
    //! \brief Seek the cursor position.
    //! \param[in] cursor  Target cursor position.
    //! \return Either \c true if possible, otherwise \c false.
    //----------------------------------------
    virtual bool seek(const int64_t cursor) = 0;

    //========================================
    //! \brief Seek cursor to begin of stream.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    virtual bool seekBegin() = 0;

    //========================================
    //! \brief Seek cursor to end of stream.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    virtual bool seekEnd() = 0;

    //========================================
    //! \brief Get the current cursor position.
    //! \return Current cursor position or -1 for EOF.
    //----------------------------------------
    virtual int64_t tell() = 0;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
