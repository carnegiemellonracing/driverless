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
//! \date Jun 5, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/StreamBase.hpp>
#include <microvision/common/sdk/io/DataPackage.hpp>
#include <microvision/common/sdk/listener/Emitter.hpp>
#include <microvision/common/sdk/misc/Utils.hpp>

#include <type_traits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Interface to provide read functionality for data sources.
//!
//! Inherit from this interface to provide functionality to read data sources.
//! Aligned to your implementation add an implementation of StreamReaderFactoryExtension.
//!
//! \extends StreamBase
//! \extends Emitter
//!
//! \sa StreamReaderFactory
//------------------------------------------------------------------------------
class DataPackageStreamReader : public virtual StreamBase, public virtual Emitter
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using StreamBase::open;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~DataPackageStreamReader() override = default;

public:
    //========================================
    //! \brief Read first DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual DataPackagePtr readFirstPackage() = 0;

    //========================================
    //! \brief Read last DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual DataPackagePtr readLastPackage() = 0;

    //========================================
    //! \brief Read next DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual DataPackagePtr readNextPackage() = 0;

    //========================================
    //! \brief Read previous DataPackage.
    //! \return Either DataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual DataPackagePtr readPreviousPackage() = 0;

    //========================================
    //! \brief Skip the next n DataPackage blocks.
    //! \param[in] packageCount  Count of packages too skip.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    virtual bool skipNextPackages(const uint32_t packageCount) = 0;

    //========================================
    //! \brief Skip the n previous DataPackage blocks.
    //! \param[in] packageCount  Count of packages too skip.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    virtual bool skipPreviousPackages(const uint32_t packageCount) = 0;

public:
    //========================================
    //! \brief Utility method that goes through the stream and notifies all registered
    //!        streamers / listeners without time synchronisation
    //! \return The number of processed messages Zero if the stream is not open/empty
    //!         or no listeners / streamers are registered.
    //----------------------------------------
    virtual uint32_t loopAndNotify();
};

//==============================================================================

//========================================
//! \brief Nullable DataPackageStreamReader pointer.
//----------------------------------------
using DataPackageStreamReaderPtr = std::unique_ptr<DataPackageStreamReader>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
