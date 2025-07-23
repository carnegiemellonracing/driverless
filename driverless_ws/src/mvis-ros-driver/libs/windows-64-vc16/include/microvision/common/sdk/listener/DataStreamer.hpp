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
//! \date Sep 3, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>
#include <microvision/common/sdk/MsgBufferBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class DataStreamer
//! \brief Abstract base class for all streamer classes.
//! \date Sep 4, 2013
//!
//! A DataStreamer can be registered to an IdcDevice to receive \b all
//! DataContainer received by that device.
//! The method onData will be called in the context of the receive thread
//! of that device.
//!
//! The data will \b not be deserialized. So a DataStreamer can be used
//! to forward via network or write data types to a file. Since the
//! associated IdcDataHeader also will be provided the same timestamp
//! of the header can left as well untouched.
//!
//! User implementation of DataStreamer can apply filter or manipulate
//! the timestamps of the IdcDataHeader.
//!
//! If one wants to manipulate the contents of a DataContainer it may be
//! better to implement a DataContainerListener.
//!
//! \sa DataContainerListener
//------------------------------------------------------------------------------
class DataStreamer
{
public:
    using ConstCharVectorPtr = MsgBufferBase::ConstCharVectorPtr;

    //========================================
    //! \brief Destrutor does nothing special.
    //----------------------------------------
    virtual ~DataStreamer() = default;

public:
    //========================================
    //! \brief Method to be called if a new DataContainer
    //!        (in serialized form) has been received.
    //! \param[in] dh    Meta data of the DataContainer.
    //! \param[in] data  Serialized DataContainer.
    //----------------------------------------
    virtual void onData(const IdcDataHeader& dh, ConstCharVectorPtr data) = 0;
}; // DataStreamer

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
