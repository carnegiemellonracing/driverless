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
//! \date Feb 6, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/Device.hpp>
#include <microvision/common/sdk/listener/IdcEmitter.hpp>

#include <microvision/common/logging/logging.hpp>

#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface which provides functionality for listeners and commands.
//!
//! \extends Device
//! \extends IdcEmitter
//! \sa DataContainerListenerBase
//------------------------------------------------------------------------------
class IdcDevice : public Device, public IdcEmitter
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcDevice() override = default;

    //========================================
    //! \brief Checks that a connection is established and packages are still processed.
    //! \returns Either \c true if connection is established or work is ongoing on received data or otherwise \c false.
    //----------------------------------------
    bool isWorking() override { return isConnected(); }
}; // class IdcDevice

//==============================================================================
//! \brief Nullable IdcDevice pointer.
//------------------------------------------------------------------------------
using IdcDevicePtr = std::unique_ptr<IdcDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
