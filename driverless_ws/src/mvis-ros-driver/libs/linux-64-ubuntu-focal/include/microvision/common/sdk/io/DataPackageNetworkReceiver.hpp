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
//!\brief Interface to provide receive functionality for network data sources.
//!
//! Inherit from this interface to provide functionality to receive from network data sources.
//! Will be implemented by UdpReceiver, for example.
//!
//! \extends NetworkBase
//------------------------------------------------------------------------------
class DataPackageNetworkReceiver : public virtual NetworkBase
{
public:
    //========================================
    //! \brief Input queue value type.
    //----------------------------------------
    using QueueValueType = DataPackagePtr;

    //========================================
    //! \brief Input queue type.
    //! \note Used the list implementation in way of the queue concept.
    //----------------------------------------
    using QueueType = std::list<QueueValueType>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~DataPackageNetworkReceiver() override = default;

public:
    //========================================
    //! \brief Get ThreadSafe::Access to the input queue where all received data will store.
    //! \returns ThreadSafe::Access to the input queue.
    //----------------------------------------
    virtual ThreadSafe<QueueType>::Access getInputQueue() = 0;
};

//==============================================================================

//========================================
//! \brief Nullable DataPackageNetworkReceiver pointer.
//----------------------------------------
using DataPackageNetworkReceiverPtr = std::unique_ptr<DataPackageNetworkReceiver>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
