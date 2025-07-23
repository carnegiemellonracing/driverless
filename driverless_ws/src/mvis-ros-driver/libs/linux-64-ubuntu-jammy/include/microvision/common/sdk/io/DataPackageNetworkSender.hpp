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
//! \date Sep 30, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/sdk/io/NetworkBase.hpp>
#include <microvision/common/sdk/io/DataPackage.hpp>

#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Interface to provide send functionality via network data sources.
//!
//! Inherit from this interface to provide functionality to send via network data sources.
//! Will implemented by UdpSender for example.
//!
//! \extends NetworkBase
//------------------------------------------------------------------------------
class DataPackageNetworkSender : public virtual NetworkBase
{
public:
    //========================================
    //! \brief Output queue value type.
    //----------------------------------------
    using QueueValueType = DataPackagePtr;

    //========================================
    //! \brief Output queue type.
    //! \note Used the list implementation in way of the queue concept.
    //----------------------------------------
    using QueueType = std::list<QueueValueType>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~DataPackageNetworkSender() override = default;

public:
    //========================================
    //! \brief Get ThreadSafe::Access to the output queue where all sendable data will store.
    //! \returns ThreadSafe::Access to the output queue.
    //----------------------------------------
    virtual ThreadSafe<QueueType>::Access getOutputQueue() = 0;
};

//==============================================================================
//! \brief Nullable DataPackageNetworkSender pointer.
//------------------------------------------------------------------------------
using DataPackageNetworkSenderPtr = std::unique_ptr<DataPackageNetworkSender>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
