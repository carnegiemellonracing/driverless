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

#include <microvision/common/sdk/io/DataPackage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface for all data package listeners.
//!
//! A DataPackageListener can be registered at a \c Device to receive \b all
//! forwarded DataPackages.
//! The method onDataReceived will be called in the context of the receive thread
//! of that Device.
//!
//! The data will \b not be deserialized into data container for example a Scan2209.
//! So a DataPackageListener can be used to forward raw payload data via network
//! or write data blocks to a file.
//! Use a DataContainerListener to be notified of deserialized data containers.
//!
//! \sa Emitter
//! \sa DataContainerListener
//! \sa IdcDataPackageListener
//------------------------------------------------------------------------------
class DataPackageListener
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~DataPackageListener() = default;

public:
    //========================================
    //! \brief Method to be called when a new DataPackage is received.
    //! \param[in] data  Shared pointer to instance of data package containing received data.
    //----------------------------------------
    virtual void onDataReceived(const DataPackagePtr& data) = 0;
}; // DataPackageListener

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
