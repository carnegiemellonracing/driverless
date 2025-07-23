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
//! \date Jul 12, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/MsgBufferBase.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>

#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <boost/asio/deadline_timer.hpp>
#include <boost/system/error_code.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief
//! \date Jul 12, 2016
//!
//!
//------------------------------------------------------------------------------
class PassiveMsgBuffer final : public MsgBufferBase
{
public:
    //========================================
    //! \brief Constructor
    //!
    //! \param[in]       bufSize     Size of the buffer
    //!                              which will be allocated
    //!                              to hold the received
    //!                              message data. This size has to be
    //!                              significantly larger than the largest
    //!                              to be expected message.
    //----------------------------------------
    explicit PassiveMsgBuffer(const int32_t bufSize);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~PassiveMsgBuffer() override = default;

public:
    bool getMessage(const IdcDataHeader*& recvDataHeader, ConstCharVectorPtr& msgBody, const int32_t nbOfBytesReceived);

protected:
    bool increaseInsPos(const int32_t nbOfBytesReceived);
}; // PassiveMsgBuffer

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
