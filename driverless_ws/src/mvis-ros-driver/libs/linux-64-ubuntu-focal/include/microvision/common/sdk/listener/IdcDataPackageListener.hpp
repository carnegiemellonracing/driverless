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
//! \date Aug 29, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/IdcDataPackage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface for all idc data package listeners.
//!
//! A IdcDataPackageListener can be registered at IdcDevice to receive \b all
//! forwarded IdcDataPackage.
//! The method onDataReceived will be called in the context of the receive thread
//! of that IdcDevice.
//!
//! The data will \b not be deserialized. So a IdcDataPackageListener can be used
//! to forward via network or write data types to a file.
//!
//! \sa DataContainerListener
//------------------------------------------------------------------------------
class IdcDataPackageListener
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~IdcDataPackageListener() = default;

public:
    //========================================
    //! \brief Method to be called if a new IdcDataPackage has been received.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    virtual void onDataReceived(const IdcDataPackagePtr& data) = 0;
}; // DataPackageListener

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
